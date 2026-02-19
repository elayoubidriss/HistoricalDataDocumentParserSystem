import os
from typing import List

from langchain_community.document_loaders import UnstructuredFileLoader


class UnstructuredPowerPointLoader(UnstructuredFileLoader):

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.file_utils.filetype import FileType, detect_filetype

        unstructured_version = tuple(
            [int(x) for x in __unstructured_version__.split(".")]
        )
        try:

            is_ppt = detect_filetype(self.file_path) == FileType.PPT
        except ImportError:
            _, extension = os.path.splitext(str(self.file_path))
            is_ppt = extension == ".ppt"

        if is_ppt and unstructured_version < (0, 4, 11):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning .ppt files is only supported in unstructured>=0.4.11. "
                "Please upgrade the unstructured package and try again."
            )

        if is_ppt:
            from unstructured.partition.ppt import partition_ppt

            return partition_ppt(filename=self.file_path, **self.unstructured_kwargs)
        else:
            from unstructured.partition.pptx import partition_pptx

            return partition_pptx(filename=self.file_path, **self.unstructured_kwargs)