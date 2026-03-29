"""Configuration dataclasses with TOML serialization."""

from __future__ import annotations

import importlib.resources
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import tomli_w


@dataclass
class GeneralConfig:
    output_dir: str = "./output"
    log_level: str = "INFO"


@dataclass
class ParsingConfig:
    ocr_enabled: bool = False
    ocr_engine: str = "easyocr"
    max_pages: int = 0
    extract_tables: bool = False
    extract_images: bool = False


@dataclass
class ChunkingConfig:
    strategy: str = "hybrid"
    max_tokens: int = 512
    overlap_tokens: int = 64
    include_metadata: bool = True


@dataclass
class MetadataConfig:
    include_page_numbers: bool = True
    include_section_titles: bool = True
    include_element_types: bool = True
    custom_tags: list[str] = field(default_factory=list)


@dataclass
class ProcessingConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    def to_toml(self, path: Path) -> None:
        """Write config to a TOML file."""
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def from_toml(cls, path: Path) -> ProcessingConfig:
        """Load config from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            general=GeneralConfig(**data.get("general", {})),
            parsing=ParsingConfig(**data.get("parsing", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            metadata=MetadataConfig(**data.get("metadata", {})),
        )

    @classmethod
    def load_defaults(cls) -> ProcessingConfig:
        """Load the default config shipped with the package."""
        defaults_path = importlib.resources.files("autorag").joinpath("defaults.toml")
        with importlib.resources.as_file(defaults_path) as path:
            return cls.from_toml(path)
