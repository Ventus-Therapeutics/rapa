#!/usr/bin/env python
"""
Main file to run rapa

Copyright (c) 2025 Ventus Therapeutics U.S., Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""

import argparse
import sys, os
import timeit
from datetime import datetime

import setup_protein as stp
import hydrogen_placement_sp2 as hsp2
import state_assignment as sa
import global_constants as gc
from misc import prettify_time, Logger, generate_multi_pdb_pymol_script

def get_unknown_list_for_pml(structure):
    all_unknown_residues = [] # this will be [('A', 253, 'HIE'), ('B', 101, 'SER')...]]
    for unknown_res in stp.get_unknown_residue_list(structure):
        chainID = unknown_res.parent.id
        res_num = unknown_res.id[1]
        all_unknown_residues.append((chainID, res_num, unknown_res.resname))
    return all_unknown_residues

def parse_arguments(argv):
    """
    process arguments and parse them to global when needed
    """
    parser = argparse.ArgumentParser(description='RAPA')
    parser.add_argument('-pID', '--protein_id',dest='protID', type=str, required=True, metavar='',
                        help='Input protein ID name. It is a required argument of data type string.')

    parser.add_argument('-o', '--out_name',dest='out_prefix', type=str, default='', metavar='',
                        help='Prefix of output file name. It is an optional argument of data type string. If not provided, it will use the default value as "[pID]_out". ')

    parser.add_argument('-s', '--single_pdb_out',
                        help='This flag requests the program to provide a single output pdb file. It is done by setting the cut off that defines degeneracy of unknown residues to zero.',
                        action="store_true")

    parser.add_argument('-l', '--loud', dest='log_file',
                        help='Print out more details and save it to a log file, default is off',
                        action='store_true')

    parser.add_argument('-d', '--debug',
                        help='Print out even more details for debugging',
                        action="store_true")

    parser.add_argument('-k_hlp', '--keep_hlp',
                        help = 'Keep the PDB created with the added hydrogen and lone pair coordinates ({prefix}_HLPsp2.pdb), Default: off (delete file after)',
                        action="store_true")

    parser.add_argument('-hlp', '--HLPsp2_known',
                        help='A flag to indicate if the additional PDB (having suffix "_HLPsp2.pdb"), with SP2 hydrogen and lone pair coordinates exist. This can occur if the user requested the program to generate this additional PDB by providing "-k_hlp" as a flag in the previous run for the same PDB. Note, it is assumed RAPA for the given PDB is needed to run twice. During the first run the user provided the "-k_hlp" flag and generated the additional PDB (with suffix "_HLPsp2.pdb"). This was then used in the second run by providing "-hlp" in the second run. This two step approach is recommended for debugging as it stores an intermediate calculation and saves computational time.',
                        action="store_true")

    parser.add_argument('--pymol',
                        help='If on, will generates a PyMOL script(.pml) that will load all the generated pdb files '
                             'and make each unknown residues a scene',
                        action='store_true')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.1.1 ')

    args = parser.parse_args(argv)

    # Protein id
    # the input can either be 1bcd or 1bcd.pdb
    # we're stripping the prefix
    args.protID, _ = os.path.splitext(args.protID)

    # set up outputs
    if args.out_prefix == '':
        args.out_prefix = args.protID

    args.output_folder = f"{args.out_prefix}_outputs"
    if os.path.isdir(args.output_folder):
        print(f"WARNING: specified output folder {args.output_folder} already exist, will overwrite")
    else:
        os.makedirs(args.output_folder)

    args.log_name = f"{args.output_folder}/{args.out_prefix}.log"

    # pass to global variable
    gc.debug = args.debug
    gc.log_file = args.log_file
    gc.out_folder = args.output_folder
    gc.out_info_file = f"{args.output_folder}/{args.out_prefix}.info"

    if args.single_pdb_out:
        gc.ECutOff = 0
    else:
        gc.ECutOff = 1

    return args

def main(argv):

    args = parse_arguments(argv)

    if gc.log_file:
        # If the log file option is on, print out info and direct stdout to file as well
        sys.stdout = Logger(args.log_name)


    starttime = timeit.default_timer()
    print('PROGRAM CALL:python {}'.format(' '.join(sys.argv)))

    # set up structure, know the number of model and chain in the pdb
    structure = stp.setup_structure(args.protID, outFolder='.', fName=None)
    chains = [item for sublist in structure for item in sublist]
    num_models = len(structure.child_list)
    num_chains = len(chains)

    if gc.log_file:
        print(f"The PDB ID being used: {args.protID} and start time is: "
              f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
        print(f"\nThe structure has number of models {num_models}: {structure.child_list} and number of chains: {num_chains}: {chains}\n")

    with open(gc.out_info_file, "w") as fInfo:
        fInfo.write(f"PDB ID: {args.protID} \n")
        fInfo.flush()
        fInfo.write(f"Models: {num_models}: {structure.child_list}\n")
        fInfo.write(f"Chains: {num_chains}: {chains}\n\n")
        fInfo.flush()

    if gc.debug:
        print(f"PDB ID: {args.protID}")
        print(f"Models: {num_models}: {structure.child_list}")
        print(f"Chains: {num_chains}: {chains}\n")


    # rename HID/HIE/HIP/ASH/GLH->HIS/HIS/HIS/ASP/GLU
    stp.change_HIDEP_ASH_GLH(structure)

    # initialize the known/unknown residues(ASN/GLN/HIS, and mark if 2 ASPs/GLUs are in h-bonding distance to each other)
    stp.set_initial_known_residues_and_rotamers(structure)

    gc.xc_orig, gc.yc_orig, gc.zc_orig = stp.get_centroid(structure)
    # normalize so the new centroid=(0,0,0)
    stp.normalize_atom_coords(structure)

    ###########################Adding hydrogen and  connected to sp2 side chain atoms of unknown residue ##########################
    if not args.HLPsp2_known:
        lastSerial = list(structure.get_residues())[-1].child_list[-1].serial_number
        # This does not use neighboring residues
        lastSerial, _ = hsp2.place_lonepair_on_backbone(structure, lastSerial)
        lastSerial, _ = hsp2.add_sp2_sidechain_lonepairs(structure, lastSerial)
        lastSerial, _ = hsp2.placeHydrogens_backbone(structure, lastSerial)
        # Adding hydrogen to side chains:
        lastSerial, _ = hsp2.add_sp2_sidechain_hydrogens(structure, lastSerial)
        fName = f'{args.output_folder}/{args.protID}_HLPsp2.pdb'
        stp.write_to_PDB(structure, fName, removeHLP=False, removeHall=False, set_original_centroid=False)
        if gc.log_file:
            print("After adding the hydrogens....")
        protIDName = args.protID + "_HLPsp2"

    else:
        protIDName = args.protID + "_HLPsp2"

        if gc.log_file:
            print("Hydrogen and lone pairs were already present!")


    # setting up the new structure with hydrogen and lone pairs
    structure = stp.setup_structure(protIDName, outFolder=args.output_folder, fName=None)
    stp.set_initial_known_residues_and_rotamers(structure)

    # flagging the unknown ASP/GLUs
    unknownASP, unknownASP_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName='ASP')
    unknownGLU, unknownGLU_atomInfo = stp.get_all_unknown_ASP_GLU(structure, resName='GLU')
    over2ASPs = 0
    over2GLUs = 0
    if (unknownASP):
        structure, uniqueIDsASP, over2ASPs = stp.accomodate_for_ASPs_GLUs(structure, unknownASP, unknownASP_atomInfo)
    if (unknownGLU):
        structure, uniqueIDsGLU, over2GLUs = stp.accomodate_for_ASPs_GLUs(structure, unknownGLU, unknownGLU_atomInfo)

    # Flag the side chain residue hydrogen position for SER, THR, LYS, TYR is unknown.
    stp.set_initial_residue_side_chain_hydrogen_unknown(structure)

    # Get all unkown residues before start resolving structures
    all_unknown_res = get_unknown_list_for_pml(structure)

    # Print out unknown residues that RAPA is dealing with
    lines = ['\n'+"#"*100, f"\nTotal number of unknown residues: {len(all_unknown_res)}\n", "Unknown residues:\n"]
    for (chain, res_id, res_name) in all_unknown_res:
        lines.append(f" {res_name}-{res_id} of chain {chain}\n")
    msg = ''.join(lines)
    with open(gc.out_info_file, "a") as fInfo:
        fInfo.write(msg)
        fInfo.flush()
    print(msg)


    generated_files, _, branched_residues = sa.resolve_residue_ambiguities_in_one_structure(structure, set_original_centroid=True,
                                                                         generated_files=None, pdb_file_num=None,
                                                                         outprefix=args.out_prefix, branched_residues=None)
    total_run_time = timeit.default_timer() - starttime
    # print out branching info
    if len(branched_residues) != 0:
        lines = ['#'*100+'\n', f'Residues w/ degenerate states: (energy of different states within '
                              f'{gc.ECutOff} kcal/mol)\n']
        for (chain, res_id, res_name) in list(set(branched_residues)):
            lines.append(f" {res_name}-{res_id} of chain {chain}\n")
        msg = ''.join(lines)
        with open(gc.out_info_file, "a") as fInfo:
            fInfo.write(msg)
            fInfo.flush()
        print(msg)

    if args.pymol:
        print("Generating pymol script based on output PDB files")
        generate_multi_pdb_pymol_script(generated_files, all_unknown_res, f'{gc.out_folder}/inspect.pml')

    lines = ['#'*100+'\n',f'Done. Execution time is: {prettify_time(total_run_time)}\n',f"Number of files generated: {len(generated_files)} and files generated:\n{'\n'.join(generated_files)}\n"]
    msg = ''.join(lines)
    with open(gc.out_info_file, "a") as fInfo:
        fInfo.write(msg)
        fInfo.flush()
    print(msg)

    if (over2ASPs > 0 or over2GLUs > 0):
        if gc.log_file:
            print(f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log_file file for details")

        with open(gc.out_info_file, "a") as fInfo:
            fInfo.write(
                f"WARNING: There are more than 2 ASPs/GLUs.\n\n over2ASPs: {over2ASPs}\n over2GLUs: {over2GLUs}. Human intervention is required. Look for WARNING in log_file file for details\n")
            fInfo.flush()

    if not args.keep_hlp:
        file_hlp = f"{args.output_folder}/{args.protID}_HLPsp2.pdb"
        if os.path.exists(file_hlp):
            os.remove(file_hlp)
        else:
            if gc.debug:
                print(f"File {file_hlp} does not exist.")

    print("************************************************************")
    print(f"RAPA exiting. Run for the given PDB: {args.protID} is completed")
    print("************************************************************")
    sys.exit()



if __name__ == "__main__":
    main(sys.argv[1:])
