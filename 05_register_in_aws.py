"""Helper to create Athena DDL statements for curated and prediction datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class AthenaTableConfig:
    """Configuration for an Athena external table."""

    table_name: str
    s3_location: str
    partitioned_by: str


CURATED_SCHEMA = {
    "flight_date": "date",
    "op_carrier": "string",
    "op_carrier_fl_num": "int",
    "origin": "string",
    "dest": "string",
    "crs_dep_time": "int",
    "dep_time": "int",
    "dep_delay": "double",
    "arr_time": "int",
    "arr_delay": "double",
    "distance": "double",
    "delayed": "int",
}

PREDICTION_SCHEMA = {
    **CURATED_SCHEMA,
    "prediction": "int",
    "prediction_proba": "double",
}


def _format_schema(schema: Dict[str, str]) -> str:
    return ",\n    ".join([f"{col} {dtype}" for col, dtype in schema.items()])


def build_ddl(config: AthenaTableConfig, schema: Dict[str, str], file_format: str = "parquet") -> str:
    """Construct an Athena DDL statement for an external table."""

    columns_sql = _format_schema(schema)
    ddl = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {config.table_name} (
    {columns_sql}
)
PARTITIONED BY ({config.partitioned_by})
STORED AS {file_format.upper()}
LOCATION '{config.s3_location}'
TBLPROPERTIES ('parquet.compression'='SNAPPY');
"""
    return ddl


def print_instructions(curated_config: AthenaTableConfig, prediction_config: AthenaTableConfig) -> None:
    """Print SQL statements and guidance for running them in Athena."""

    curated_ddl = build_ddl(curated_config, CURATED_SCHEMA)
    predictions_ddl = build_ddl(prediction_config, PREDICTION_SCHEMA)

    logger.info("Run the following statements in Athena to register your tables:")
    logger.info("Curated table DDL:\n%s", curated_ddl)
    logger.info("Predictions table DDL:\n%s", predictions_ddl)
    logger.info(
        "After creating tables, MSCK REPAIR TABLE or AWS Glue crawler can be used to load partitions."
    )


__all__ = ["AthenaTableConfig", "print_instructions", "build_ddl", "CURATED_SCHEMA", "PREDICTION_SCHEMA"]
