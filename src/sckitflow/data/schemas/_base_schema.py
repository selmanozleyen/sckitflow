import abc

__all__ = ["StrictDataSchema"]


class StrictDataSchema(abc.ABC):
    """Base class for the data schemas: each validates its arguments at construction (:meth:`_verify_args`).

    A schema only describes the data; the loaders read it, and :class:`~sckitflow.data.DataManager` sizes the model
    from the representations' shapes.
    """

    def __init__(self) -> None:
        """Initializes the class and verifies the arguments."""
        self._verify_args()

    @abc.abstractmethod
    def _verify_args(self) -> None:
        """Verifies the validity of arguments set at initialization."""
        raise NotImplementedError
