import sys

def generate_multi_pdb_pymol_script(pdb_files, residues, output_pml="multi_pdb_show_residues.pml"):
    """
    generate a pymol session that loads all generated pdb files with each unknown residues being made into a scene
    """
    with open(output_pml, 'w') as f:
        f.write("# Auto-generated PyMOL script for multiple PDBs\n\n")

        # Load all PDBs and assign object names
        for i, pdb in enumerate(pdb_files):
            obj_name = f"model_{i}"
            f.write(f"load {pdb}, {obj_name}\n")
        f.write("set label_size, 20\n")
        f.write("bg_color white\n")
        f.write("cartoon loop\n")
        f.write("hide everything, all\n")
        f.write("show cartoon, all\n")
        f.write("\n")

        # For each residue, show it across all models and store one scene
        for j, (chain, resi, _) in enumerate(residues):
            scene_name = f"res_{chain}_{resi}"

            f.write(f"# Residue {chain}:{resi} across all PDBs\n")
            obj_name = f"model"
            sel_name = f"{obj_name}_res_{chain}_{resi}"
            near_name = f"{sel_name}_near"

            # Select residue and surroundings in this model
            f.write(f"select {sel_name}, chain {chain} and resi {resi}\n")
            f.write(f"select {near_name}, byres ({sel_name} around 4)\n")
            f.write(f"show sticks, {sel_name}\n")
            f.write(f"show sticks, {near_name}\n")

            # Create a combined selection for zooming and scene
            f.write(f"color green, * and symbol C\n")
            f.write(f"color purple, {sel_name} and symbol C\n")
            f.write(f"zoom {sel_name}\n")
            f.write(f"scene {scene_name}, store\n")
            f.write(f"hide sticks, {sel_name}\n")
            f.write(f"hide sticks, {near_name}\n\n")
            f.write(f"delete {sel_name}\n")
            f.write(f"delete {near_name}\n")

        f.write("# End of script\n")


class Logger(object):
    """
    Logger class that can simultaneously write what's printed to log file
    """

    def __init__(self,logname):
        self.terminal = sys.stdout
        self.log = open(logname, "w",buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def prettify_time(seconds):
    """
    Converts a time in seconds to a more human-readable format, handling hours, minutes, and seconds.

    Parameters:
        seconds (float): The time in seconds.

    Returns:
        str: The time in a prettified format (hours, minutes, seconds, etc.).
    """
    if seconds >= 3600:  # More than 1 hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif seconds >= 60:  # More than 1 minute
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    elif seconds >= 1:  # More than 1 second
        return f"{seconds:.6f} seconds"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} milliseconds"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} microseconds"
    else:
        return f"{seconds * 1e9:.3f} nanoseconds"
