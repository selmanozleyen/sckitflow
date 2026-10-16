from anndata import AnnData

from sckitflow.data.containers._coupling import CouplingData, StateData
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(StrictDataSchema):
    """Data schema implementing the logic for coupling data.

    Coupling data is intended as the observation representation that will be used
    to match source and target distributions. Whenever these are not specified,
    it will default to the state representation, which is the space on which
    generative modeling will be done.
    """

    def __init__(
        self,
        rep_lin: str | None = None,
        source_rep_quad: str | None = None,
        target_rep_quad: str | None = None,
    ) -> None:
        """Initializes the coupling data schema.

        :param rep_lin: Optional string identifier denoting the linear representation for
            shared by source and target distribution samples. It should be a key appearing in the `.obsm`
            of annotated data objects. Defaults to `None`, in which case the attribute `.X` will be used.
            This location will identify the shared axes between the source and the target spaces.
        :type rep_lin: class: `str | None`

        :param source_rep_quad: Optional string identifier denoting the representation for the quadratic term
            of the source distribution samples. It should be a key appearing in the `.obsm`
            of annotated data objects. This loc will contain the features defined only on the source space.
        :type source_rep_quad: class: `str | None`

        :param target_rep_quad: Optional string identifier denoting the representation for the quadratic term
            of the target distribution samples. It should be a key appearing in the `.obsm`
            of annotated data objects. This loc will contain the features defined only on the target space.
        :type target_rep_quad: class: `str | None`
        """
        self._rep_lin = rep_lin
        self._source_rep_quad = source_rep_quad
        self._target_rep_quad = target_rep_quad
        super().__init__()

    def _verify_args(self) -> None:
        """No-op needed for compatibility with parent class."""
        pass

    def _verify_schema(self, adata: AnnData) -> None:
        """Verifies the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        # verify target representation
        if self._rep_lin is not None:
            self._check_key_found_in_adata_field(adata, self._rep_lin, "obsm")
        if self._source_rep_quad is not None:
            self._check_key_found_in_adata_field(adata, self._source_rep_quad, "obsm")
        if self._target_rep_quad is not None:
            self._check_key_found_in_adata_field(adata, self._target_rep_quad, "obsm")

    def _get_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        # Collect linear term; this will be shared between source and target.
        X_lin = StateData(self._extract_array(adata, repr=self._rep_lin))
        # Quadratic terms for source and target, collected separately (specific dimensions).
        if self._source_rep_quad is not None:
            X_source_quad = StateData(self._extract_array(adata, repr=self._source_rep_quad))
        else:
            X_source_quad = None
        if self._target_rep_quad is not None:
            X_target_quad = StateData(self._extract_array(adata, repr=self._target_rep_quad))
        else:
            X_target_quad = None
        # Construct the coupling data.
        source_coupling = CouplingData(state_lin=X_lin, state_quad=X_source_quad)
        target_coupling = CouplingData(state_lin=X_lin, state_quad=X_target_quad)
        return source_coupling, target_coupling

    @property
    def rep_lin(self) -> str | None:
        """Exposes to `rep_lin` parameter set at initialization."""
        return self._rep_lin

    @property
    def source_rep_quad(self) -> str | None:
        """Exposes to `source_rep_quad` parameter set at initialization."""
        return self._source_rep_quad

    @property
    def target_rep_quad(self) -> str | None:
        """Exposes to `target_rep_quad` parameter set at initialization."""
        return self._target_rep_quad
