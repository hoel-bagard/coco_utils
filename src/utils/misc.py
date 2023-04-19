from shutil import get_terminal_size


def clean_print(msg: str, fallback: tuple[int, int] = (156, 38), end: str = "\n") -> None:
    r"""Function that prints the given string to the console and erases any previous print made on the same line.

    Args:
        msg: String to print to the console
        fallback: Size of the terminal to use if it cannot be determined by shutil (if using windows for example)
        end: What to add at the end of the print. Usually '\n' (new line), or '\r' (back to the start of the line)
    """
    print(msg + " " * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)
