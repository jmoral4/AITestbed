#!/usr/bin/env python3
import os
import argparse
import sys
import re # Using re for more flexible input splitting

def find_files_by_extension(directory, extensions=None, recursive=False, excluded_dirs=None):
    """Finds all files with specified extensions in the directory.

    Args:
        directory (str): The path to the directory to search.
        extensions (list, optional): List of file extensions to search for (e.g., ['.cs', '.py']).
                                    Defaults to ['.cs'] if None.
        recursive (bool): Whether to search subdirectories.
        excluded_dirs (list, optional): A list of directory names to exclude.
                                        Defaults to None (no exclusions).

    Returns:
        list: A sorted list of full paths to the found files.
              Returns an empty list if the directory is invalid.
    """
    found_files = []
    if extensions is None:
        extensions = ['.cs']

    # Normalize extensions to lowercase and ensure they start with a dot
    normalized_extensions = []
    for ext in extensions:
        ext = ext.lower().strip()
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext)
    if excluded_dirs is None:
        excluded_dirs = []

    # Normalize excluded directory names: lowercase and stripped of surrounding slashes/backslashes.
    # e.g., "\obj\", "obj/", "Obj" all become "obj"
    normalized_excluded_dirs = []
    for d in excluded_dirs:
        # Remove leading/trailing path separators (both / and os.sep)
        # and convert to lowercase for case-insensitive comparison.
        norm_d = d.lower()
        if norm_d.startswith(os.sep):
            norm_d = norm_d[len(os.sep):]
        if norm_d.endswith(os.sep):
            norm_d = norm_d[:-len(os.sep)]
        if norm_d.startswith('/'):
            norm_d = norm_d[1:]
        if norm_d.endswith('/'):
            norm_d = norm_d[:-1]
        if norm_d: # Avoid adding empty strings if input was just "/" or "\"
            normalized_excluded_dirs.append(norm_d)

    # Remove duplicates that might arise from normalization (e.g., "obj" and "obj/")
    normalized_excluded_dirs = sorted(list(set(normalized_excluded_dirs)))


    if not os.path.isdir(directory):
        print(f"Error: Directory not found or is not a valid directory: {directory}", file=sys.stderr)
        return found_files # Return empty list

    ext_display = ', '.join(normalized_extensions)
    print(f"Searching for files with extensions [{ext_display}] in '{directory}'{' recursively' if recursive else ''}...")
    if normalized_excluded_dirs:
        print(f"Excluding directory names: {', '.join(normalized_excluded_dirs)}")


    if recursive:
        for root, dirs, files in os.walk(directory, topdown=True):
            # Prune the list of directories to visit
            # Exclude any directory whose name (lowercase) is in normalized_excluded_dirs
            dirs[:] = [d for d in dirs if d.lower() not in normalized_excluded_dirs]

            for filename in files:
                filename_lower = filename.lower()
                if any(filename_lower.endswith(ext) for ext in normalized_extensions):
                    full_path = os.path.join(root, filename)
                    found_files.append(full_path)
    else:
        # Non-recursive search: exclusions apply to subdirectories,
        # so they don't directly affect file listing in the immediate directory.
        try:
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                item_lower = item.lower()
                if os.path.isfile(full_path) and any(item_lower.endswith(ext) for ext in normalized_extensions):
                    found_files.append(full_path)
        except OSError as e:
            print(f"Error accessing directory {directory}: {e}", file=sys.stderr)
            return [] # Return empty list on access error

    found_files.sort() # Sort for consistent ordering
    return found_files

def find_files_in_directories(
        directories: list[str],
        extensions=None,
        recursive=False,
        excluded_dirs=None) -> list[str]:
    """Finds all files with specified extensions in multiple directories.

    Args:
        directories (list[str]): List of directory paths to search.
        extensions (list, optional): List of file extensions to search for (e.g., ['.cs', '.py']).
                                    Defaults to ['.cs'] if None.
        recursive (bool): Whether to search subdirectories.
        excluded_dirs (list, optional): A list of directory names to exclude.
                                        Defaults to None (no exclusions).

    Returns:
        list: A sorted list of unique full paths to the found files.
    """
    all_files: set[str] = set()
    for d in directories:
        files = find_files_by_extension(
                    d, extensions, recursive, excluded_dirs)
        all_files.update(files)
    return sorted(all_files)

def select_files(file_list, search_directory):
    """Presents the list of files to the user and gets their selection.

    Args:
        file_list (list): A list of file paths.
        search_directory (str): The initial directory that was searched.
                                Used for displaying relative paths.

    Returns:
        list: A list of file paths selected by the user.
              Returns an empty list if no valid selection is made.
    """
    if not file_list:
        print("No files found with specified extensions.")
        return []

    print("\nFound files:")
    for i, f_path in enumerate(file_list):
        try:
            # Display path relative to the initial search directory
            display_path = os.path.relpath(f_path, start=search_directory)
        except ValueError: # Handle cases like different drives or if f_path is not under search_directory
             display_path = f_path
        print(f"  {i + 1}: {display_path}")

    selected_files = []
    while True:
        try:
            raw_input = input("\nEnter the numbers of the files to include (e.g., 1 3 5 or 2,4,6), or 'all' to select all: ")
            raw_input = raw_input.strip().lower()

            if raw_input == 'all':
                 print("Selected all files.")
                 return file_list # Return all found files

            parts = re.split(r'[,\s]+', raw_input)
            parts = [p for p in parts if p] # Filter out empty strings

            if not parts:
                 print("No selection entered. Please try again.")
                 continue

            selected_indices = set()
            valid_input = True
            for part in parts:
                if not part.isdigit():
                    print(f"Error: '{part}' is not a valid number.")
                    valid_input = False
                    break
                num = int(part)
                if 1 <= num <= len(file_list):
                    selected_indices.add(num - 1)
                else:
                    print(f"Error: Number {num} is out of range (1-{len(file_list)}).")
                    valid_input = False
                    break

            if not valid_input:
                continue

            if not selected_indices:
                print("No valid file numbers were selected. Please try again.")
                continue

            sorted_indices = sorted(list(selected_indices))
            selected_files = [file_list[i] for i in sorted_indices]

            print("\nSelected files:")
            for f in selected_files:
                 try:
                     display_path = os.path.relpath(f, start=search_directory)
                 except ValueError:
                     display_path = f
                 print(f"  - {display_path}")
            print("-" * 30)

            confirm = input("Confirm selection? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected_files
            else:
                print("Selection cancelled. Please enter numbers again.")

        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces or commas.")
        except EOFError:
             print("\nOperation cancelled by user.")
             return []
        except KeyboardInterrupt:
             print("\nOperation cancelled by user.")
             return []


def _relative_to_roots(path: str, roots: list[str]) -> str:
    """Get relative path from a file to the best matching root directory.

    Args:
        path (str): Absolute or relative path to the file.
        roots (list[str]): List of root directory paths.

    Returns:
        str: Relative path with root directory name as prefix,
             or absolute path as fallback.
    """
    abs_path = os.path.abspath(path)
    for root in roots:
        try:
            rel = os.path.relpath(abs_path, start=root)
            # If rel does not start with '..' we found the right root
            if not rel.startswith('..'):
                return os.path.join(os.path.basename(root), rel)
        except ValueError:
            pass
    return abs_path        # fallback – unusual but safe

def create_context_content(selected_files, root_directories):
    """Creates context content from selected files as a string.

    Args:
        selected_files (list): List of full paths to the files to include.
        root_directories (list[str] | str): The root directories that were searched,
                                           or a single directory for backwards compatibility.
                                           Used for displaying relative paths.

    Returns:
        str: The combined content of all files with headers.
    """
    if isinstance(root_directories, str):
        root_directories = [root_directories]       # legacy path
    if not selected_files:
        return ""

    content_lines = []
    for f_path in selected_files:
        try:
            relative_path = _relative_to_roots(f_path, root_directories)
            header = f"--------------\n{relative_path}\n--------------"
            content_lines.append(header)

            with open(f_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                content_lines.append(content)
                content_lines.append("")  # Empty line separator

        except FileNotFoundError:
            print(f"Warning: File not found during reading: {f_path}. Skipping.", file=sys.stderr)
            relative_path = _relative_to_roots(f_path, root_directories)
            content_lines.append(f"Error: File not found - {relative_path}")
            content_lines.append("")
        except IOError as e:
            print(f"Warning: Could not read file {f_path}: {e}. Skipping.", file=sys.stderr)
            relative_path = _relative_to_roots(f_path, root_directories)
            content_lines.append(f"Error reading {relative_path}: {e}")
            content_lines.append("")
        except Exception as e:
            print(f"Warning: An unexpected error occurred processing file {f_path}: {e}. Skipping.", file=sys.stderr)
            relative_path = _relative_to_roots(f_path, root_directories)
            content_lines.append(f"Error processing {relative_path}: {e}")
            content_lines.append("")

    return "\n".join(content_lines)


def create_context_file(selected_files, root_directories, output_filename="prompt_context.txt"):
    """Combines the content of selected files into a single output file.

    Args:
        selected_files (list): List of full paths to the files to include.
        root_directories (list[str] | str): The root directories that were searched,
                                           or a single directory for backwards compatibility.
                                           Used for displaying relative paths.
        output_filename (str): The name of the output file.
    """
    if not selected_files:
        print("No files were selected. Exiting.")
        return

    print(f"\nCreating context file: {output_filename}...")

    # Get the context content using the reusable function
    context_content = create_context_content(selected_files, root_directories)

    if not context_content:
        print("No content to write. Exiting.")
        return

    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.write(context_content)

        print(f"\nSuccessfully created '{output_filename}' with content from {len(selected_files)} file(s).")

    except IOError as e:
        print(f"Error: Could not write to output file {output_filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: An unexpected error occurred while writing the output file: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Combine source code files into a single context file for AI prompts."
    )
    parser.add_argument("directories", nargs='+',
                        help="The directories to search for files.")
    parser.add_argument("-x", "--extensions",
                        action="append",
                        default=[],
                        metavar="EXT",
                        help="File extension to include (e.g., .cs, .py, .js). "
                             "Can be used multiple times. Defaults to .cs if none specified.")
    parser.add_argument("-r", "--recursive",
                        action="store_true",
                        help="Search directories recursively.")
    parser.add_argument("-o", "--output",
                        default="prompt_context.txt",
                        help="Name of the output context file "
                             "(default: prompt_context.txt)")
    parser.add_argument("-e", "--exclude",
                        action="append",
                        default=[],
                        metavar="DIR_NAME",
                        help="Directory name to exclude (e.g., obj, bin, .git). "
                             "Can be used multiple times.")

    # ----------  test hook  ----------
    if len(sys.argv) == 1:                        # no CLI arguments supplied
        sys.argv.extend([r"C:\git\pathfinderai", "-r", "-e", "obj", "-x", ".cs"])    # <- test values
    # ---------------------------------

    args = parser.parse_args()

    # Use specified extensions or default to .cs
    extensions = args.extensions if args.extensions else ['.cs']

    found_files = find_files_in_directories(args.directories, extensions, args.recursive, args.exclude)

    if not found_files:
        # find_files_in_directories prints specific errors if directories are invalid.
        # If directories are valid but no files found, select_files will handle it.
        invalid_dirs = [d for d in args.directories if not os.path.isdir(d)]
        if invalid_dirs: # Check if error was due to invalid directories
            sys.exit(1) # Indicate failure
        # Otherwise, it might be that no files matched or all were excluded.
        # select_files will print "No files found with specified extensions."

    # Pass first directory for backward compatibility in selection prompt
    selected_files_list = select_files(found_files, args.directories[0])

    if selected_files_list:
        create_context_file(selected_files_list, args.directories, args.output)
    else:
        print("No files selected or process cancelled. Output file not created.")
        sys.exit(0) # Indicate successful exit, but nothing done


if __name__ == "__main__":
    main()
