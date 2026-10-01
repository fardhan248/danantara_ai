from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from unstructured.chunking.title import chunk_by_title
import utils.contextmanager_utils as cm

from collections import defaultdict
import fitz, re, uuid, pdfplumber, math, base64, copy
from typing import IO
from io import BytesIO
from PIL import Image

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
        self.chunks = None

        self._elements = None
        self._all_tables = None
        self._text_chunks = None

    async def _create_elements(self):
        if self.filename:
            self._elements = partition_pdf(
                filename=self.filename,
                strategy="hi_res",
                languages=["ind", "eng"],
                infer_table_structure=True,
            )
        else:
            self._elements = partition_pdf(
                file=self.filebytes,
                strategy="hi_res",
                languages=["ind", "eng"],
                infer_table_structure=True,
            )

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
        self._all_tables = [self.chunks[i] for i in range(len(self.chunks)) if self.chunks[i].to_dict()["type"] in ["Table", "TableChunk"]]
        self._text_chunks = [self.chunks[i] for i in range(len(self.chunks)) if self.chunks[i].to_dict()["type"] == "CompositeElement"]

    async def _search_table_context(self):
        chunks = defaultdict(list)
        for table in self._all_tables:
            page_numbers = table.metadata.page_numbers
            for page_number in page_numbers:
                for chunk in self._text_chunks:
                    if page_number in chunk.metadata.page_numbers:
                        chunks[table.metadata.chunk_id].append(chunk.text)

        return chunks

    async def _build_batch_messages_table(self):
        system_prompt = """You are a table desciption generator for table retriever based on the table description vector embedding.
Based on the given table, image description, and contexts, explain detail about the table CONTENTS and its MEANING, NOT the structure.

contexts:
{contexts}

table:
{table}

HIGHLY IMPORTANT NOTE:
- DO NOT EXPLAIN the table structure.
- Do not mention table structural elements such as number of rows/column, header/column names, or phrases like "Tabel ini terdiri dari..."
- Do not write opening sentence, just immediately describe the table.
- Do not halucinate when generating table description. JUST USE BASED ON THE GIVEN CONTEXTS, if the table and the contexts are related.
- Use Indonesian language for the description.
- Strictly maximum 800 characters."""

        chunks = await self._search_table_context()

        batch_messages = []
        for table in self._all_tables:
            chunk_id = table.metadata.chunk_id
            tab = getattr(table.metadata, "text_as_html", table.text)
            
            context = chunks.get(chunk_id, [])

            system_msg = SystemMessage(
                system_prompt.format_map({"contexts": context, "table": tab})
            )

            batch_messages.append([system_msg, HumanMessage(content="Describe the table in Indonesia language.")])

        return batch_messages

    async def _create_table_description(self):
        batch_messages = await self._build_batch_messages_table()

        responses = await self.client.abatch(batch_messages, config={"max_concurrency": 5})

        for i, table in enumerate(self._all_tables):
            table.metadata.description = responses[i].content
        
    async def start(self):
        # Extraction
        await self._create_elements()
        await self._chunk_elements()
        await self._create_table_description()
