#tools/builtin/read_file.py
from pydantic import BaseModel, Field

from utils.path import resolve_path, is_binary_file
from utils.text import count_tokens, truncate_text
from tools.base import Tool, ToolConformation, ToolInvocation, ToolKind, ToolResult


class ReadFileParams(BaseModel):
    path: str = Field(
        ...,
        description="Path to the file to read (relative to working directory or absolute)",
    )
    offset: int = Field(
        1, ge=1, description="Line number to start reading from(1-based). Defaluts to 1"
    )
    limit: int | None = Field(
        None,
        ge=1,
        description="Maximum number of lines to read. If not specified, reads entire file.",
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the content of a text file. Returns the file content with line numbers."
        "For large files, use offset and limit to read specific portions."
        "Cannot read binary files (images, executables, etc.)"
    )
    kind = ToolKind.READ
    schema = ReadFileParams

    MAX_FILE_SIZE = 1024 * 1024 * 10
    MAX_OUTPUT_TOKENS = 25000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        # check path
        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")
        if not path.is_file():
            return ToolResult.error_result(f"Path is not a file: {path}")

        # check size
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(
                f"File too large ({file_size / (1024*1024):.1f}MB)"
                f"Maximum capacity is {self.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        # check binary
        if is_binary_file(path):
            file_size_mb = file_size / (1024 * 1024)
            size_str = (
                f"{file_size_mb:.2f}MB" if file_size_mb >= 1 else f"{file_size} bytes"
            )
            return ToolResult.error_result(
                (
                    f"Cannot read binary file: {path.name} | ({size_str})"
                    f"This tool only reads text files."
                )
            )

        try:
            # read file
            try:
                content = path.read_text(encoding="utf-8")
            except:
                content = path.read_text(encoding="latin-1")

            # from file to list[str]
            lines = content.splitlines()
            total_lines = len(lines)
            if total_lines == 0:
                return ToolResult.success_result(
                    "file is empty,",
                    metadata={
                        "lines": total_lines,
                    },
                )

            # reading start_idx to end_idx
            start_idx = max(0, params.offset - 1)
            if params.limit is not None:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            # from list[str] to str
            selected_lines = lines[start_idx:end_idx]
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted_lines.append(f"{i:6} | {line}")
            output = "\n".join(formatted_lines)

            # truncate as per token count
            token_count = count_tokens(output, "cl100k_base")
            truncated = False
            if token_count > self.MAX_OUTPUT_TOKENS:
                output = truncate_text(
                    output,
                    self.MAX_OUTPUT_TOKENS,
                    suffix=f"\n... [truncated {total_lines} total lines]",
                )
                truncated = True

            # prepare file + metadata
            metadata_lines = []
            if start_idx > 0 or end_idx < total_lines:
                metadata_lines.append(
                    f"Showing lines {start_idx+1}-{end_idx} of {total_lines}"
                )
            if metadata_lines:
                header = " | ".join(metadata_lines) + "\n\n"
                output = header + output

            # return in format
            return ToolResult.success_result(
                output=output,
                truncated=truncated,
                # Think what metadata will be needed for user and for llm
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "shown_start": start_idx + 1,
                    "shown_end": end_idx,
                },
            )

            # error handling
        except Exception as e:
            return ToolResult.error_result(f"Failed to read file: {e}")
