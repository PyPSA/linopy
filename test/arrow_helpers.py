"""A table known by the Arrow PyCapsule stream interface alone, as a DuckDB relation or a pyarrow RecordBatchReader."""

from __future__ import annotations

from typing import Any

import polars as pl


class ArrowStream:
    """*data* exported once through ``__arrow_c_stream__``, as a stream that cannot be read twice."""

    def __init__(self, data: pl.DataFrame | pl.Series) -> None:
        self.data = data
        self.exported = False

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        assert not self.exported, "an Arrow stream is read once"
        self.exported = True
        return self.data.__arrow_c_stream__(requested_schema)
