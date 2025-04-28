#!/usr/bin/env python3
import os
import argparse
import sys
import re # Using re for more flexible input splitting

def find_cs_files(directory, recursive=False):
    """Finds all .cs files in the specified directory.

    Args:
        directory (str): The path to the directory to search.
        recursive (bool): Whether to search subdirectories.

    Returns:
        list: A sorted list of full paths to the found .cs files.
              Returns an empty list if the directory is invalid.
    """
    cs_files = []
    if not os.path.isdir(directory):
        print(f"Error: Directory not found or is not a valid directory: {directory}", file=sys.stderr)
        return cs_files # Return empty list

    print(f"Searching for .cs files in '{directory}'{' recursively' if recursive else ''}...")

    if recursive:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(".cs"):
                    full_path = os.path.join(root, filename)
                    cs_files.append(full_path)
    else:
        try:
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                if os.path.isfile(full_path) and item.lower().endswith(".cs"):
                    cs_files.append(full_path)
        except OSError as e:
            print(f"Error accessing directory {directory}: {e}", file=sys.stderr)
            return [] # Return empty list on access error

    cs_files.sort() # Sort for consistent ordering
    return cs_files

def select_files(file_list):
    """Presents the list of files to the user and gets their selection.

    Args:
        file_list (list): A list of file paths.

    Returns:
        list: A list of file paths selected by the user.
              Returns an empty list if no valid selection is made.
    """
    if not file_list:
        print("No .cs files found.")
        return []

    print("\nFound .cs files:")
    for i, f_path in enumerate(file_list):
        # Show path relative to the initial search directory if possible, else full path
        try:
            # Find common prefix to potentially shorten displayed paths
            # This assumes the first file's directory is a reasonable base
            base_dir = os.path.dirname(file_list[0])
            display_path = os.path.relpath(f_path, start=os.path.commonpath([base_dir, f_path]))
        except ValueError: # Handle cases like different drives on Windows
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

            # Use regex to split by comma or space, handling multiple spaces/commas
            parts = re.split(r'[,\s]+', raw_input)
            # Filter out empty strings resulting from multiple delimiters
            parts = [p for p in parts if p]

            if not parts:
                 print("No selection entered. Please try again.")
                 continue

            selected_indices = set() # Use a set to automatically handle duplicates
            valid_input = True
            for part in parts:
                if not part.isdigit():
                    print(f"Error: '{part}' is not a valid number.")
                    valid_input = False
                    break
                num = int(part)
                if 1 <= num <= len(file_list):
                    selected_indices.add(num - 1) # Store 0-based index
                else:
                    print(f"Error: Number {num} is out of range (1-{len(file_list)}).")
                    valid_input = False
                    break

            if not valid_input:
                continue # Ask for input again

            if not selected_indices:
                print("No valid file numbers were selected. Please try again.")
                continue

            # Sort indices to maintain order similar to the displayed list
            sorted_indices = sorted(list(selected_indices))
            selected_files = [file_list[i] for i in sorted_indices]

            print("\nSelected files:")
            for f in selected_files:
                 try:
                     base_dir = os.path.dirname(file_list[0])
                     display_path = os.path.relpath(f, start=os.path.commonpath([base_dir, f]))
                 except ValueError:
                     display_path = f
                 print(f"  - {display_path}")
            print("-" * 30) # Separator

            confirm = input("Confirm selection? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected_files
            else:
                print("Selection cancelled. Please enter numbers again.")
                # Loop continues to re-prompt

        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces or commas.")
        except EOFError: # Handle Ctrl+D
             print("\nOperation cancelled by user.")
             return []
        except KeyboardInterrupt: # Handle Ctrl+C
             print("\nOperation cancelled by user.")
             return []


def create_context_file(selected_files, output_filename="prompt_context.txt"):
    """Combines the content of selected files into a single output file.

    Args:
        selected_files (list): List of full paths to the files to include.
        output_filename (str): The name of the output file.
    """
    if not selected_files:
        print("No files were selected. Exiting.")
        return

    print(f"\nCreating context file: {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for f_path in selected_files:
                try:
                    # Use basename for the header
                    filename_header = os.path.basename(f_path)
                    header = f"--------------\n{filename_header}\n--------------\n"
                    outfile.write(header)

                    with open(f_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        # Add a newline after content unless it already ends with one
                        if not content.endswith('\n'):
                             outfile.write('\n\n')
                        else:
                             outfile.write('\n') # Add one extra newline for spacing

                except FileNotFoundError:
                    print(f"Warning: File not found during writing: {f_path}. Skipping.", file=sys.stderr)
                except IOError as e:
                    print(f"Warning: Could not read file {f_path}: {e}. Skipping.", file=sys.stderr)
                except Exception as e:
                     print(f"Warning: An unexpected error occurred processing file {f_path}: {e}. Skipping.", file=sys.stderr)


        print(f"\nSuccessfully created '{output_filename}' with content from {len(selected_files)} file(s).")

    except IOError as e:
        print(f"Error: Could not write to output file {output_filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: An unexpected error occurred while writing the output file: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Combine C# (.cs) files into a single context file for AI prompts.")
    parser.add_argument("directory",
                        help="The directory to search for .cs files.")
    parser.add_argument("-r", "--recursive",
                        action="store_true",
                        help="Search directories recursively.")
    parser.add_argument("-o", "--output",
                        default="prompt_context.txt",
                        help="Name of the output context file (default: prompt_context.txt)")

    args = parser.parse_args()

    found_files = find_cs_files(args.directory, args.recursive)

    if not found_files:
        # find_cs_files prints specific errors, just exit cleanly
        sys.exit(1) # Indicate failure

    selected_files_list = select_files(found_files)

    if selected_files_list:
        create_context_file(selected_files_list)
    else:
        print("No files selected or process cancelled. Output file not created.")
        sys.exit(0) # Indicate successful exit, but nothing done


if __name__ == "__main__":
    main()