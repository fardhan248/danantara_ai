from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from unstructured.chunking.title import chunk_by_title
import utils.contextmanager_utils as cm

from collections import defaultdict
import fitz, re, uuid, pdfplumber, math, base64, copy
from typing import IO
from io import BytesIO

async def _create_image_message(img_base64: str | list[str], prompt: str):
    if isinstance(img_base64, str):
        img_base64 = [img_base64]

    content = [{"type": "text", "text": prompt}]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in img_base64
    ]

    return HumanMessage(content=content)

class ExtractPDF:
    def __init__(
            self, 
            filetype: str,
            client: ChatOpenAI,
            knowledge_id: str,
            filename: str | None = None, 
            filebytes: IO[bytes] | None = None,
    ):
        self.filename = filename
        self.filebytes = filebytes
        self.filetype = filetype
        self.client = client

        if self.filename is None and self.filebytes is None:
            raise ValueError("There is must be a filename or filebytes.")

        self.knowledge_id = knowledge_id
        self.tables = None
        self.len_doc = None
        self.images_out_table = None
        self.images_in_table = None
        self.chunks = None

        self._page_height = None
        self._elements = None
        self._all_images = None
        self._all_tables = None

    async def _create_elements(self):
        if self.filename:
            self._elements = partition_pdf(
                filename=self.filename,
                strategy="fast",
                languages=["ind", "eng"],
            )
        else:
            self._elements = partition_pdf(
                file=self.filebytes,
                strategy="fast",
                languages=["ind", "eng"],
            )

    @staticmethod
    async def table_to_markdown(tab):
        tab = [[re.sub(r"\n", " ", c) if c is not None else "" for c in t] for t in tab]
        header = tab[0]
        rows = tab[1:]

        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(c) for c in row) + " |\n"
        return md

    async def _create_tables(self):
        if self.filename:
            file_type = self.filename
        else: 
            file_type = self.filebytes

        all_tables = [] # [{"table_id": str, "table": str, "page_number": int, "coord": ()}]
        with pdfplumber.open(file_type) as pdf:
            self.len_doc = len(pdf.pages)
            self._page_height = pdf.pages[0].height

            for i, page in enumerate(pdf.pages):
                tabs = page.find_tables({"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"})

                if tabs:
                    for tab in tabs:
                        coord = tab.bbox # (x0, y0, x1, y1)
                        all_tables.append({
                            "table_id": str(uuid.uuid4()),
                            "page_number": i + 1,
                            "table": tab.extract(),
                            "coord": coord,
                            "object": tab,
                        })

        self._all_tables = all_tables

    async def _create_images(self):
        all_images = [] #[{"img_path": str, "page_number": int, "coord": ()}]

        if self.filename:
            pdf = fitz.open(self.filename)
        else:
            pdf = fitz.open(stream=self.filebytes, filetype=self.filetype)

        for i, page in enumerate(pdf):
            page_w, page_h = page.rect[2], page.rect[3]
            page_center_x, page_center_y = page_w/2, page_h/2
            images = page.get_images(full=True)

            # akan diperiksa apakah watermark atau bukan, 
            # dengan heuristik: ukuran height = width dan posisi di tengah page
            for img in images:
                coord = page.get_image_rects(img)[0]
                img_center_x = (coord[0] + coord[2]) / 2
                img_center_y = (coord[1] + coord[3]) / 2

                img_w, img_h = img[2], img[3]

                if (img_w == img_h) and (math.isclose(page_center_x, img_center_x, abs_tol=0.5) and math.isclose(page_center_y, img_center_y, abs_tol=0.5)):
                    continue
                else:
                    # Upload to MinIO
                    xref = img[0]
                    base_img = pdf.extract_image(xref)

                    img_bytes = base_img["image"]
                    buf = BytesIO(img_bytes)
                    size = len(img_bytes)

                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                    image_id = str(uuid.uuid4())
                    destination_file = f"{self.knowledge_id}/image_{image_id}_{xref}.{base_img['ext']}"
                    bucket_name = "image-bucket"

                    try:
                        client = cm.minio
                        found = await client.bucket_exists(bucket_name)
                        if not found:
                            await client.make_bucket(bucket_name)

                        await client.put_object(
                            bucket_name,
                            destination_file,
                            buf,
                            size,
                            content_type=f"image/{base_img['ext']}"
                        )
                    except Exception as e:
                        raise ValueError("Error upload file to MinIO.")

                    # Append to list
                    all_images.append({
                        "image": img_base64,
                        "img_path": destination_file,
                        "page_number": i + 1,
                        "coord": coord,
                        "image_id": image_id,
                    })

        pdf.close()

        self._all_images = all_images

    async def _filter_element_overlap_table(self):
        new_elements = []
        for element in self._elements:
            metadata = element.metadata
            page_number = metadata.page_number
            coord = metadata.coordinates.points

            x0, y0 = coord[0]
            x1, y1 = coord[2]

            overlap_found = False
            for table in self._all_tables: # [{"table_id": str, "table": str, "page_number": int, "coord": ()}]
                if page_number == table["page_number"]:
                    x0_tab, y0_tab, x1_tab, y1_tab = table["coord"]

                    if (x0_tab <= x0) and (y0_tab <= y0) and (x1 <= x1_tab) and (y1 <= y1_tab):
                        overlap_found = True # Skip if overlap
                        break

            if not overlap_found:
                new_elements.append(element)

        self._elements = new_elements

    async def _filter_image(self):
        images_out_table = []
        images_in_table = []

        for image in self._all_images:
            page_number = image["page_number"]
            x0, y0, x1, y1 = image["coord"]

            matched_table = None

            for table in self._all_tables:
                if page_number == table["page_number"]:
                    x0_tab, y0_tab, x1_tab, y1_tab = table["coord"]

                    # Cek titik gambar kiri-atas di dalam tabel
                    if (x0_tab <= x0 <= x1_tab) and (y0_tab <= y0 <= y1_tab):
                        matched_table = table
                        break

            if matched_table is not None:
                tab = matched_table["object"]
                table_id = matched_table["table_id"]

                matched_cell = None
                for i, row in enumerate(tab.rows):
                    for j, col in enumerate(row.cells):
                        if col:
                            x0_cell, y0_cell, x1_cell, y1_cell = col
                            tol = 8
                            if (x0_cell - tol <= x0 <= x1_cell + tol) and (y0_cell - tol <= y0 <= y1_cell + tol):
                                matched_cell = (i, j)
                                break
                    if matched_cell:
                        break

                if matched_cell:
                    i, j = matched_cell
                    image["table_id"] = table_id
                    image["row"] = i
                    image["column"] = j

                    matched_table["table"][i][j] += " [GAMBAR]"

                images_in_table.append(image)
            else:
                images_out_table.append(image)

        self.images_out_table = images_out_table
        self.images_in_table = images_in_table

    async def _combine_table(self):
        tables = [] # [{"table_id": str, "table": str, "page_numbers": []}]
        for i in range(len(self._all_tables)):
            tab = self._all_tables[i]
            start_pos = tab["coord"][1]
            page_number = tab["page_number"]

            if tables:
                prev_tab = self._all_tables[i - 1]
                prev_page_number = prev_tab["page_number"]

                same_page = (page_number == prev_page_number)
                next_page = (page_number == prev_page_number + 1)

                if same_page:
                    # gap = start_pos - prev_tab["coord"][3]
                    is_close = False #0 <= gap < 23
                elif next_page:
                    last_tab_end_pos = self._page_height - prev_tab["coord"][3]
                    gap = last_tab_end_pos + start_pos
                    is_close = gap < 2 * 95
                else:
                    is_close = False

                if not is_close:
                    tables.append({
                        "table_id": tab["table_id"],
                        "table_ids": [tab["table_id"]],
                        "page_numbers": [page_number],
                        "table": tab["table"],
                        "begin_coord": tab["coord"],
                        "image_ids": [],
                    })
                else:
                    tables[-1]["table"] += tab["table"]
                    tables[-1]["table_ids"].append(tab["table_id"])
                    tables[-1]["page_numbers"].append(tab["page_number"])

            else:
                tables.append({
                    "table_id": tab["table_id"],
                    "table_ids": [tab["table_id"]],
                    "page_numbers": [page_number],
                    "table": tab["table"],
                    "begin_coord": tab["coord"],
                    "image_ids": [],
                })

        for i in range(len(tables)):
            tables[i]["table"] = await self.table_to_markdown(tables[i]["table"])

        self.tables = tables

    async def _chunk_elements(self):
        chunks = chunk_by_title(self._elements, include_orig_elements=True, max_characters=800, new_after_n_chars=640, combine_text_under_n_chars=800)

        for chunk in chunks:
            chunk.metadata = copy.deepcopy(chunk.metadata)
            chunk.metadata.chunk_id = str(uuid.uuid4())
            page_numbers = []
            for element in chunk.metadata.orig_elements:
                page_number = element.metadata.page_number
                if page_number not in page_numbers:
                    page_numbers.append(page_number)
                
            chunk.metadata.page_numbers = page_numbers

        self.chunks = chunks

    async def _input_chunk_id(self, chunk_id, end_first_element, start_second_element, page_first_element, page_second_element):
        # Same page
        if page_first_element == page_second_element:
            diff = start_second_element - end_first_element
            if 0 <= diff < 23:
                return chunk_id
            else:
                return ""

        # Different page
        else:
            gap = (self._page_height - end_first_element) + start_second_element
            if gap < 2*95:
                return chunk_id
            else:
                return ""
        
    async def _search_image_out_table_parent_text(self):
        # Image with parent of text
        for image in self.images_out_table:
            image_page_number = image["page_number"]
            found = False
            for chunk in self.chunks:
                if found:
                    break
                chunk_id = chunk.metadata.chunk_id
                for element in chunk.metadata.orig_elements:
                    element_page_number = element.metadata.page_number

                    if (image_page_number == element_page_number) or ((image_page_number - 1) == element_page_number):
                        element_coord = element.metadata.coordinates.points
                        end_of_element = element_coord[2][1] # y1
                        start_of_image = image["coord"][1] # y0

                        result = await self._input_chunk_id(chunk_id, end_of_element, start_of_image, element_page_number, image_page_number)

                        if result:
                            image["chunk_id"] = result
                            found = True
                            break

            if not found:
                image["chunk_id"] = ""
                        
    async def _search_image_out_table_parent_image(self):
        # Image with parent of other image
        for image in self.images_out_table:
            if not image["chunk_id"]:
                image_page_number = image["page_number"]
                found = False
                for image2 in self.images_out_table:
                    if found:
                        break
                    image2_page_number = image2["page_number"]
                    chunk_id = image2.get("chunk_id", "")
                    if (image["img_path"] != image2["img_path"]) and ((image_page_number == image2_page_number) or ((image_page_number - 1) == image2_page_number)):
                        end_of_image2 = image2["coord"][3]
                        start_of_image = image["coord"][1]

                        result = await self._input_chunk_id(chunk_id, end_of_image2, start_of_image, image2_page_number, image_page_number)
                        if result:
                            image["chunk_id"] = result
                            found = True
                            break

                if not found:
                    image["chunk_id"] = ""

    async def _search_table_parent(self):
        for tab in self.tables:
            tab_page_number = tab["page_numbers"][0]

            found = False
            # Table with parent of text
            for chunk in self.chunks:
                if found:
                    break
                chunk_id = chunk.metadata.chunk_id
                for element in chunk.metadata.orig_elements:
                    element_page_number = element.metadata.page_number

                    if (tab_page_number == element_page_number) or ((tab_page_number - 1) == element_page_number):
                        element_coord = element.metadata.coordinates.points
                        end_of_element = element_coord[2][1] # y1
                        start_of_tab = tab["begin_coord"][1] # y0

                        result = await self._input_chunk_id(chunk_id, end_of_element, start_of_tab, element_page_number, tab_page_number)
                        if result:
                            tab["chunk_id"]= result
                            found = True
                            break

            if not found:
                found = False
                # Table with parent of image
                for image in self.images_out_table:
                    image_page_number = image["page_number"]
                    chunk_id = image.get("chunk_id", "")

                    if (tab_page_number == image_page_number) or ((tab_page_number - 1) == image_page_number):
                        end_of_image = image["coord"][3]
                        start_of_tab = tab["begin_coord"][1]

                        result = await self._input_chunk_id(chunk_id, end_of_image, start_of_tab, image_page_number, tab_page_number)
                        if result:
                            tab["chunk_id"] = result
                            found = True
                            break

            if not found:
                tab["chunk_id"] = ""

    async def _trace_image_in_table(self):
        for image in self.images_in_table:
            found = False
            for tab in self.tables:
                chunk_id = tab["chunk_id"]
                if image["table_id"] in tab["table_ids"]:
                    image["chunk_id"] = chunk_id
                    tab["image_ids"].append(image["image_id"])
                    found = True
                    break

            if not found:
                image["chunk_id"] = ""

    async def _metadata_chunks(self):
        for chunk in self.chunks:
            # image out table:
            chunk.metadata.image_out_table = []
            for image in self.images_out_table:
                if image.get("chunk_id", "") == chunk.metadata.chunk_id:
                    image_id = image.get("image_id") 
                    if image_id not in chunk.metadata.image_out_table:
                        chunk.metadata.image_out_table.append(image_id)

            # image in table:
            chunk.metadata.image_in_table = []
            for image in self.images_in_table:
                if image.get("chunk_id", "") == chunk.metadata.chunk_id:
                    image_id = image.get("image_id") 
                    if image_id not in chunk.metadata.image_in_table:
                        chunk.metadata.image_in_table.append(image_id)

            # table:
            chunk.metadata.table = []
            for table in self.tables:
                if table.get("chunk_id", "") == chunk.metadata.chunk_id:
                    table_id = table.get("table_id")
                    if table_id not in chunk.metadata.table:
                        chunk.metadata.table.append(table_id)

    async def _search_image_context(self, images):
        imgs = []
        image_ids = []
        contexts = defaultdict(list)
        tables = defaultdict(list)
        for image in images:
            img_base64 = image['image']
            imgs.append(img_base64)
            image_ids.append(image["image_id"])

            # search for contexts
            for chunk in self.chunks:
                if image["chunk_id"] == chunk.metadata.chunk_id:
                    contexts[image["image_id"]].append(chunk.text)

            # search for tables
            for table in self.tables:
                if image["chunk_id"] == table["chunk_id"]:
                    tables[image["image_id"]].append(table["table"])

        return imgs, image_ids, contexts, tables
    
    async def _build_batch_messages(self, images):
        system_prompt = """You are an image desciption generator for image retriever based on the image description vector embedding.
Based on the given image and contexts, explain detail about the CONTENTS or PURPOSE of the image (based on the contexts), NOT the elements.

contexts:
{contexts}

tables (if any):
{tables}

HIGHLY IMPORTANT NOTE:
- DO NOT explain the image elements.
- Do not write opening sentence, just immediately describe the image.
- Do not halucinate when generating image description. JUST USE BASED ON THE GIVEN CONTEXTS, if the image and the contexts are related.
- Use Indonesian language for the description.
- Strictly maximum 800 characters."""

        imgs, image_ids, contexts, tables = await self._search_image_context(images)

        batch_messages = []
        for img, image_id in zip(imgs, image_ids):
            context = contexts.get(image_id, [])
            table = tables.get(image_id, [])

            system_msg = SystemMessage(
                system_prompt.format_map({"contexts": context, "tables": table})
            )
            human_msg = await _create_image_message(img, "Describe this image in Indonesia language.")

            batch_messages.append([system_msg, human_msg])

        return batch_messages

    async def _create_image_description(self):
        batch_out_table = await self._build_batch_messages(self.images_out_table)
        batch_in_table = await self._build_batch_messages(self.images_in_table)

        all_batch_messages = batch_out_table + batch_in_table

        responses = await self.client.abatch(all_batch_messages, config={"max_concurrency": 5})

        n_out = len(batch_out_table)
        response_out_table = responses[:n_out]
        response_in_table = responses[n_out:]

        for i, image in enumerate(self.images_out_table):
            image["description"] = response_out_table[i].content

        for i, image in enumerate(self.images_in_table):
            image["description"] = response_in_table[i].content

    async def _search_table_context(self):
        chunks = defaultdict(list)
        images_out_table = defaultdict(list)
        images_in_table = defaultdict(list)
        for table in self.tables:
            chunk_id = table.get("chunk_id", "")
            if chunk_id:
                # Chunk
                for chunk in self.chunks:
                    if chunk_id == chunk.metadata.chunk_id:
                        chunks[chunk_id].append(chunk.text)

                # Image out table
                for img in self.images_out_table:
                    if chunk_id == chunk.metadata.chunk_id:
                        images_out_table[chunk_id].append(img["description"])

                # Image in table
                for img in self.images_in_table:
                    if chunk_id == img["chunk_id"]:
                        desc = f"{img['description']}\nrow, column: {img['row']}, {img['column']}"
                        images_in_table[chunk_id].append(desc)

        return chunks, images_out_table, images_in_table

    async def _build_batch_messages_table(self):
        system_prompt = """You are a table desciption generator for table retriever based on the table description vector embedding.
Based on the given table, image description, and contexts, explain detail about the table CONTENTS and its MEANING, NOT the structure.

contexts:
{contexts}

table:
{table}

image description inside table:
{img_in}

image description outside table:
{img_out}

HIGHLY IMPORTANT NOTE:
- DO NOT EXPLAIN the table structure.
- Do not mention table structural elements such as number of rows/column, header/column names, or phrases like "Tabel ini terdiri dari..."
- Do not write opening sentence, just immediately describe the table.
- Do not halucinate when generating table description. JUST USE BASED ON THE GIVEN CONTEXTS, if the table and the contexts are related.
- Use Indonesian language for the description.
- Strictly maximum 800 characters."""

        chunks, images_out_table, images_in_table = await self._search_table_context()

        batch_messages = []
        for table in self.tables:
            chunk_id = table.get("chunk_id", "")
            tab = table["table"]
            
            context = chunks.get(chunk_id, [])
            image_out = images_out_table.get(chunk_id, [])
            image_in = images_in_table.get(chunk_id, [])

            system_msg = SystemMessage(
                system_prompt.format_map({"contexts": context, "table": tab, "img_in": image_in, "img_out": image_out})
            )

            batch_messages.append([system_msg, HumanMessage(content="Describe the table in Indonesia language.")])

        return batch_messages

    async def _create_table_description(self):
        batch_messages = await self._build_batch_messages_table()

        responses = await self.client.abatch(batch_messages, config={"max_concurrency": 5})

        for i, table in enumerate(self.tables):
            table["description"] = responses[i].content
        
    async def start(self):
        # Extraction
        await self._create_elements()
        await self._create_tables()
        await self._create_images()

        # Filetring
        await self._filter_element_overlap_table()
        await self._filter_image()

        # Combining
        await self._combine_table()
        await self._chunk_elements()

        # Searching
        await self._search_image_out_table_parent_text()
        await self._search_image_out_table_parent_image()
        await self._search_table_parent()
        await self._trace_image_in_table()

        await self._metadata_chunks()
        await self._create_image_description()
        await self._create_table_description()