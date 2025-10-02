import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from atomworks.ml.databases.dataclasses import DataSource
from atomworks.ml.databases.utils import _smart_cast

DATA_SOURCE_DB_PATH = "/projects/ml/datahub/experimental_data/experimental_data_sources.csv"
BACKUP_DIR = "/projects/ml/datahub/experimental_data/experimental_data_sources_backups/"


def get_data_source_db(data_source_db_path: str | None = None, create_if_not_exists: bool = False) -> pd.DataFrame:
    """
    Get the data source database.
    """
    if data_source_db_path is None:
        data_source_db_path = DATA_SOURCE_DB_PATH

    if not os.path.exists(data_source_db_path):
        if create_if_not_exists:
            pd.DataFrame(columns=DataSource.get_field_info().keys()).to_csv(data_source_db_path, index=False)
        else:
            raise FileNotFoundError(f"Data source database located at {data_source_db_path} does not exist.")

    return pd.read_csv(data_source_db_path, na_values=None)


def get_data_source(data_source_id: str) -> DataSource | None:
    """
    Get a data source from the database.
    """
    data_source_db = get_data_source_db()
    if data_source_id not in data_source_db["data_source_id"].values:
        return None
    data_source_dict = data_source_db.loc[data_source_db["data_source_id"] == data_source_id].to_dict(orient="records")[
        0
    ]
    del data_source_dict["data_source_id"]  # recreated at runtime

    # smart cast everything back to correct dtypes
    for key, value in data_source_dict.items():
        data_source_dict[key] = _smart_cast(value, DataSource.get_field_info()[key])

    return DataSource(**data_source_dict)


def upload_data_source(data_source: DataSource) -> None:
    """
    Upload a data source to the database. Will raise ValueError if the data_source_id already exists in the database.
    """
    if get_data_source(data_source.data_source_id):
        raise ValueError(
            f"Data source {data_source.data_source_id} already exists in database located at {DATA_SOURCE_DB_PATH}."
        )

    data_source_db = get_data_source_db()
    data_source_dict = asdict(data_source)

    # convert dicts to json strings so they are serializable
    for key, value in data_source_dict.items():
        if isinstance(value, dict):
            data_source_dict[key] = json.dumps(value)

    data_source_db = pd.concat([data_source_db, pd.DataFrame([data_source_dict])])
    backup_db()
    data_source_db.to_csv(DATA_SOURCE_DB_PATH, index=False)


def update_data_source(data_source: DataSource) -> None:
    """
    Update a data source in the database. Will raise ValueError if the data_source_id does not exist in the database.
    """
    if not get_data_source(data_source.data_source_id):
        raise ValueError(
            f"Data source {data_source.data_source_id} does not exist in database located at {DATA_SOURCE_DB_PATH}."
        )

    data_source_db = get_data_source_db()
    data_source_dict = asdict(data_source)
    # convert dicts to json strings so they are serializable
    for key, value in data_source_dict.items():
        if isinstance(value, dict):
            data_source_dict[key] = json.dumps(value)

    data_source_db.loc[data_source_db["data_source_id"] == data_source.data_source_id] = pd.DataFrame(
        [data_source_dict]
    )
    backup_db()
    data_source_db.to_csv(DATA_SOURCE_DB_PATH, index=False)


def backup_db() -> None:
    """
    Create a time-stamped backup of the database

    We can add smarter logic later but for now lets make sure we don't accidentally lose the whole thing somehow
    """
    timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    new_path = os.path.join(BACKUP_DIR, os.path.basename(DATA_SOURCE_DB_PATH)).replace(".csv", timestamp + ".csv")

    shutil.copy2(DATA_SOURCE_DB_PATH, new_path)
