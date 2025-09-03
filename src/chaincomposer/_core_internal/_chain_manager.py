from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING
)

from .._logging import get_logger, WARNING

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ._chain_wrapper import _ChainWrapper

class _ChainManager:
    """Manages a sequence of chain operations.

    ChainManager is a class that manages a sequence of chain operations, allowing for
    the addition of chains and running them in sequence with provided data.

    Methods:
        __init__(logger: Logger | None):
            Initializes the ChainManager instance with an optional logger.
            Type: logging.Logger | None
            Defaults to None.

        __str__() -> str:
            Returns a string representation of the ChainManager instance.

        __repr__() -> str:
            Returns a string representation of the ChainManager instance.

        add_chain(chain_wrapper, output_passthrough_key_name):
            Adds a chain to the chain sequence.

            Args:
                chain_wrapper (ChainWrapper): The chain wrapper to add
                output_passthrough_key_name (str | None): Optional output key name
                    Defaults to None.

        run(data_dict, data_dict_update_function):
            Runs the chain sequence with the provided data dictionary and optional update function.

            Args:
                data_dict (dict): The input data dictionary
                    Type: Dict[str, Any]
                data_dict_update_function (callable | None): Optional update function
                    Type: Callable[[Dict[str, Any]], None]
                    Defaults to None.
            Returns:
                dict: The updated data dictionary
                    Type: Dict[str, Any]
    """

    def __init__(
        self,
        enable_logging: bool | None = False,
        level: int | None = WARNING,
    ) -> None:
        """Initializes the ChainManager instance.

        Args:
            logger (Logger | None): A logger instance to be used for logging.
                Type: logging.Logger | None
                If not provided, a default logger will be created.
                Defaults to None.

        Attributes:
            logger (Logger): The logger instance used for logging.
                Type: logging.Logger
            chain_sequence (list): A list to store the chain sequence.
                Type: List[Tuple[_ChainWrapper, str | None]]
        """
        self.logger = get_logger(
            module_name=__name__,
            level=level,
            null_logger=not enable_logging,
        )

        self.chain_sequence: List[Tuple[_ChainWrapper, Optional[str]]] = []

    def __str__(self) -> str:
        """Returns a string representation of the ChainManager object.

        The string representation includes the index and the wrapper of each
        element in the chain_sequence.

        Returns:
            str: A string representation of the ChainManager object.
        """
        chain_info = ", ".join(
            [
                f"{idx}: {wrapper}"
                for idx, (wrapper, _) in enumerate(self.chain_sequence)
            ]
        )
        return f"ChainManager(chains=[{chain_info}])"

    def __repr__(self) -> str:
        """Returns a string representation of the object for debugging purposes.

        This method calls the __str__() method to provide a human-readable
        representation of the object.

        Returns:
            str: A string representation of the object.
        """
        return self.__str__()

    def add_chain(
        self,
        *,
        chain_wrapper: _ChainWrapper,
        output_passthrough_key_name: str | None = None,
    ) -> None:
        """Adds a chain to the chain sequence.

        Args:
            chain_wrapper (ChainWrapper): The chain wrapper to be added.
            output_passthrough_key_name (str | None): The key name for output passthrough.
                Type: str | None
                Defaults to None.

        Returns:
            None
        """
        self.chain_sequence.append((chain_wrapper, output_passthrough_key_name))

    def run(
        self,
        *,
        data_dict: Dict[str, Any],
        data_dict_update_function: Callable[[Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        """Executes a sequence of chains, updating the provided data dictionary with the results of each chain.

        Args:
            data_dict (dict): The initial data dictionary containing input variables for the chains.
                Type: Dict[str, Any]
            data_dict_update_function (callable | None): An optional function to update the data
                dictionary after each chain execution.
                Type: Callable[[Dict[str, Any]], None]
                Defaults to None.

        Returns:
            dict: The updated data dictionary after all chains have been executed.
                Type: Dict[str, Any]

        Raises:
            UserWarning: If the provided data dictionary is empty.

        Notes:
            - The method iterates over a sequence of chains (`self.chain_sequence`), executing
              each chain and updating the `data_dict` with the output.
            - If an `output_name` is provided for a chain, the output is stored in `data_dict`
              under that name; otherwise, it is stored under the key `_last_output`.
            - If `data_dict_update_function` is provided, it is called with the updated
              `data_dict` after each chain execution.
        """
        if not data_dict:
            warnings.warn(
                "No variables provided for the chain. Please ensure you have provided the necessary variables. If you have variable placeholders and do not pass them in it will result in an error."
            )

        num_chains: int = len(self.chain_sequence)
        for index, (chain_wrapper, output_name) in enumerate(self.chain_sequence):
            is_last_chain: bool = index == num_chains - 1
            output: Union[BaseModel, Dict[str, Any], str] = chain_wrapper.run_chain(
                input_data=data_dict, is_last_chain=is_last_chain
            )

            # Update data with the output
            if output_name:
                data_dict[output_name] = output
            else:
                data_dict["_last_output"] = output

            if data_dict_update_function:
                data_dict_update_function(data_dict)

        return data_dict