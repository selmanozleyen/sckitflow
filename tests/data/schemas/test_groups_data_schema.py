from collections.abc import Collection

import pytest

from sckitflow.data._group_encoders import OneHotEncoderConfig
from sckitflow.data.schemas import GroupsDataSchema

from ..shared import verify_categorical_data  # noqa

inval_key: str = "invalid_key"


ENCODER = OneHotEncoderConfig(categories=("a", "b"))


class TestGroupsDataSchema:
    def _is_valid_args(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, OneHotEncoderConfig] | None,
    ) -> bool:
        if groups is None and (groups_reps is not None or groups_encoding is not None):
            return False
        elif groups is not None and (groups_reps is None and groups_encoding is None):
            return False
        elif groups is not None and groups_reps is not None and groups_encoding is not None:
            shared_keys = set(groups_reps.keys()).intersection(set(groups_encoding.keys()))
            all_keys = set(groups_reps.keys()).union(set(groups_encoding.keys()))
            if len(shared_keys) > 0 or all_keys != set(groups):
                return False
        elif groups is not None and groups_reps is not None and groups_encoding is None:
            all_keys = set(groups_reps.keys())
            if all_keys != set(groups):
                return False
        elif groups is not None and groups_reps is None and groups_encoding is not None:
            all_keys = set(groups_encoding.keys())
            if all_keys != set(groups):
                return False
        return True

    @pytest.mark.parametrize(
        "groups",
        [
            None,
            ["source_split"],
        ],
    )
    @pytest.mark.parametrize(
        "groups_reps",
        [None, {"source_split": "source_split"}, {inval_key: "source_split"}],
    )
    @pytest.mark.parametrize(
        "groups_encoding",
        [None, {"source_split": ENCODER}, {inval_key: ENCODER}],
    )
    def test_init(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, OneHotEncoderConfig] | None,
    ) -> None:
        is_valid_args = self._is_valid_args(
            groups,
            groups_reps,
            groups_encoding,
        )
        if not is_valid_args:
            with pytest.raises(ValueError):
                _schema = GroupsDataSchema(
                    groups=groups,
                    groups_reps=groups_reps,
                    groups_encoding=groups_encoding,
                )
            return None
        _schema = GroupsDataSchema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
        )
