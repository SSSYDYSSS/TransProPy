def print_boxed_text(title):
    """
    Prints a title in a boxed format.

    This function creates a box around the given title text using hash (#) and
    equals (=) symbols. It prints the title with a border on the top and bottom,
    making it stand out in the console output.

    Parameters:
    - title: str, the text to be displayed inside the box.

    Returns:
    None. This function directly prints the formatted title to the console.
    """
    # Create the top and bottom border line of the box.
    # The border line consists of a hash symbol, followed by equals symbols
    # the length of the title plus two (for padding), and then another hash symbol.
    border_line = "#" + "=" * (len(title) + 2) + "#"

    # Print the top border line.
    print("\n" + border_line)

    # Print the title, surrounded by hash symbols and padded with one space on each side.
    print(f"# {title} #")

    # Print the bottom border line.
    print(border_line)
