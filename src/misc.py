import sys, os

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

# PDB related functions
def replace_column_range(original_line, new_value, start_col, end_col):
    """
    Replaces a specific column range (1-indexed) with a new value,
    right-aligned within that range.
    """
    # Convert 1-based PDB indexing to 0-based Python indexing
    start_idx = start_col - 1
    end_idx = end_col

    # Calculate the width of the target "slot"
    width = end_idx - start_idx

    # Format the new value:
    # '>' means right-align, 'width' sets the padding
    formatted_value = f"{str(new_value)[:width]:>{width}}"

    # Reconstruct the line
    new_line = original_line[:start_idx] + formatted_value + original_line[end_idx:]

    return new_line


def process_conect_line(line, id_map):
    """
    Processes a single CONECT line and returns a new line with updated IDs.

    Parameters:
    - line (str): The original CONECT string.
    - id_map (dict): {old_id: new_id} mapping.
    """
    # PDB CONECT columns (1-indexed):
    # Center Atom (7-11), then 4 Bonded Atoms (12-16, 17-21, 22-26, 27-31)
    column_ranges = [(7, 11), (12, 16), (17, 21), (22, 26), (27, 31)]

    new_line = list(line)  # Convert to list for easier character-level replacement

    for start, end in column_ranges:
        # Slicing the string to get the specific ID slot
        raw_val = line[start - 1:end].strip()

        if raw_val:
            old_id = int(raw_val)
            # Look up the new ID. If it's not in the map, means that ID does not need to be fixed
            new_id = id_map.get(old_id, old_id)

            # Format to right-justified string with the exact column width
            width = end - start + 1
            formatted_id = f"{new_id:>{width}}"

            # Replace the characters in the specific range
            new_line[start - 1:end] = list(formatted_id)
    return "".join(new_line)

def clean_pdb_atom_id(pdb_file):
    """
    the input PDB usually comes from MOE
    MOE will add an atom ID to TER record at the end of each chain
    and the subsequent atom ID will all continue from that TER record
    Bio PDB does not do that so the index will be different after TER record
    and CONECT record will be invalid in that case
    we should clean it by re-numbering it starting from 1 and do not assign ID to TER record
    """
    atom_id = 1
    changed_id_dict = {} # key: ori_id, value: new_id
    file_prefix = pdb_file.split('.pdb')[0]
    new_pdb_name = f'{file_prefix}_cleaned_id.pdb'
    new_pdb_f = open(new_pdb_name, 'w')
    # this is for HETATM and ANISOU will be
    with open(pdb_file, 'r') as f:
        for lc, line in enumerate(f):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # col 7-11 is atom ID
                ori_atom_id = int(line[7-1:11].strip())
                if ori_atom_id != atom_id:
                    new_line = replace_column_range(line, atom_id, 7, 11)
                    changed_id_dict[ori_atom_id] = atom_id
                    new_pdb_f.write(new_line)
                else:
                    new_pdb_f.write(line)
                atom_id += 1
            elif line.startswith("ANISOU"):
                ori_atom_id = int(line[7 - 1:11].strip())
                try:
                    new_line = replace_column_range(line, changed_id_dict[ori_atom_id], 7, 11)
                    new_pdb_f.write(new_line)
                except KeyError:
                    # if that atom_id is not in the dict that means it doesn't need to be changed
                    new_pdb_f.write(line)
            elif line.startswith("CONECT"):
                new_line = process_conect_line(line, changed_id_dict)
                new_pdb_f.write(new_line)
            else:
                new_pdb_f.write(line)
    new_pdb_f.close()
    return os.path.basename(new_pdb_name), new_pdb_name


def put_pdb_record_back(cleaned_id_pdb_file, generated_pdbs):
    """
    Bio.PDB will ignore any CONECT record, we need to append them back
    """
    # first get the CONECT record from the original PDB
    conect_lines = []
    header_lines = []
    anisou_lines = []
    anisou_atom_ids = []
    with open(cleaned_id_pdb_file, 'r') as f:
        for lc, line in enumerate(f):
            if line.startswith("CONECT"):
                conect_lines.append(line)
            elif line.startswith("HEADER"):
                header_lines.append(line)
            elif line.startswith("ANISOU"):
                anisou_lines.append(line)
                atom_id = int(line[7 - 1:11].strip())
                anisou_atom_ids.append(atom_id)
    for i in generated_pdbs:
        lines = []
        # deal with anisou record first since it needs to be following ATOM record
        with open(i, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    lines.append(line)
                    atom_id = int(line[7 - 1:11].strip())
                    if atom_id in anisou_atom_ids:
                        line_index = anisou_atom_ids.index(atom_id)
                        anisou_line = anisou_lines[line_index]
                        atom_name = line[13 -1:16]
                        residue_name = line[18-1:20]
                        new_line = replace_column_range(anisou_line, atom_name, 13, 16)
                        new_line = replace_column_range(new_line, residue_name, 18, 20)
                        lines.append(new_line)
                else:
                    lines.append(line)

        # remove END
        if lines[-1].startswith("END"):
            lines = lines[:-1]

        with open(i, "w") as out:
            out.writelines(header_lines)
            out.writelines(lines)
            out.writelines(conect_lines)
            out.write("END\n")