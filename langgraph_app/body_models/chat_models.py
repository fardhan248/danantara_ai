from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Union, Literal

class MediaInput(BaseModel):
    """
    Satu file gambar/video. Isi salah satu: `base64` atau `url`.
    - base64: isi file dalam base64 (boleh juga data URI, mis. "data:image/png;base64,...."), `mime_type` wajib
    - url: URL publik http(s) yang langsung mengarah ke file (akan diunduh, link YouTube tidak didukung)
    Video dikirim ke model sebagai beberapa frame gambar (llama.cpp tidak menerima input video langsung)
    """
    mime_type: str | None = Field(default=None, examples=["image/png", "video/mp4"])
    base64: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def check_source(self):
        # Data URI -> pisahkan mime_type dan base64
        if self.base64 and self.base64.startswith("data:"):
            header, _, data = self.base64.partition(",")
            self.mime_type = self.mime_type or header[5:].split(";")[0] or None
            self.base64 = data

        if bool(self.base64) == bool(self.url):
            raise ValueError("Isi salah satu dari 'base64' atau 'url'")
        if self.base64 and not self.mime_type:
            raise ValueError("'mime_type' wajib diisi jika menggunakan 'base64'")
        return self

class ChatInput(BaseModel):
    input_prompt: str
    sector: str
    thread_id: str | None = None
    ticker: str | None = None
    images: list[MediaInput] | None = None
    videos: list[MediaInput] | None = None

    @field_validator("images", "videos")
    @classmethod
    def check_mime_type(cls, media: list[MediaInput] | None, info):
        prefix = "image/" if info.field_name == "images" else "video/"
        for item in media or []:
            if item.mime_type and not item.mime_type.startswith(prefix):
                raise ValueError(f"mime_type '{item.mime_type}' tidak valid untuk {info.field_name}, harus '{prefix}*'")
        return media

class SummaryResumeInput(BaseModel):
    thread_id: str
    start_date: str
    end_date: str
    approved: bool
    sector: str
    ticker: str | None = None


class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str