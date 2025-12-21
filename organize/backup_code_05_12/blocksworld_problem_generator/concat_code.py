#!/usr/bin/env python3
import os
import glob

OUTPUT_FILE = "finishing_reward_function.txt"

def get_py_files():
    """
    Returns a list of all .py files in cwd, excluding this script itself.
    """
    all_py = glob.glob("*.py")
    this_script = os.path.basename(__file__)
    return [f for f in all_py if f != this_script]

def concatenate(files, out_path):
    with open(out_path, "w", encoding="utf-8") as out:
        for fname in files:
            out.write(f"The file {fname} code is this:\n")
            # out.write(f"Please understand how it is compatible with the rest of my codebase for the mission i asked you.\n")

            try:
                with open(fname, "r", encoding="utf-8") as inp:
                    out.write(inp.read())
            except Exception as e:
                out.write(f"[Error reading {fname}: {e}]\n")
            out.write(f"\n")
            # out.write(f"End: The file {fname} code is this:\n")
            out.write("\n" + "-" * 80 + "\n\n")

def main():
    infra_files = [
        "config.py",
        "state.py",
        "actions.py",
        "goal_archetypes.py",
        "backward_generator.py",
        "pddl_writer.py",
        "baseline_planner.py",
        "metadata_store.py",
        "validator.py",
        "main.py",
        "__init__.py",
        "example_usage.py",
    ]

    if not infra_files:
        print("No Python files found to concatenate.")
        return
    concatenate(infra_files, OUTPUT_FILE)
    print(f"✅ Concatenated {len(infra_files)} file(s) into '{OUTPUT_FILE}'")

    # py_files = get_py_files()
    # for p in py_files:
    #     print(f"{p}")

if __name__ == "__main__":
    main()



