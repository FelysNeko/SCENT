# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License

import hashlib
import os
import re
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Union

from rdkit import Chem


def make_dir(rel_path, *args, **kwargs):
    os.makedirs(os.path.abspath(rel_path), *args, **kwargs)


def sanitize_smi_name_for_file(smi: str):
    """
    Sanitization for file names. Replacement values cannot be part of valid SMILES.
    """
    return hashlib.sha224(smi.encode()).hexdigest()


def move_files_from_dir(source_dir_path: str, dest_dir_path: str):
    files = os.listdir(source_dir_path)
    for file in files:
        source_file = os.path.join(source_dir_path, file)
        destination_file = os.path.join(dest_dir_path, file)
        shutil.move(source_file, destination_file)


def split_list(lst, n):
    """Split a list into n equal parts, with any remainder added to the last split."""
    if n <= 0:
        raise ValueError("Number of splits (n) must be a positive integer.")

    quotient, remainder = divmod(len(lst), n)
    splits = [
        lst[i * quotient + min(i, remainder) : (i + 1) * quotient + min(i + 1, remainder)]
        for i in range(n)
    ]
    return splits


class TimedProfiler:
    def __init__(self) -> None:
        self._count = 0
        self._total = 0
        self._average = 0

    def _add_value(self, value):
        self._total += value
        self._count += 1
        self._average = self._total / self._count

    def time_it(self, fn, *args, **kwargs):
        start_time = time.time()
        res = fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
        return res

    def get_average(self):
        return self._average


class VinaDocking:
    def __init__(
        self,
        vina_cmd: str,
        receptor_pdbqt_file: str,
        center_pos: List[float],
        size: List[float],
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: int = None,
        additional_vina_args: Dict[str, str] = {},
        preparator: callable = None,
        vina_cwd: str = None,
        gpu_ids: Union[int, List[int]] = 0,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_vina_output: bool = False,
        debug: bool = False,
    ) -> None:
        """
            Parameters:
            - vina_cmd: Command line prefix to execute vina command (e.g. "/path/to/qvina2.1")
            - receptor_pdbqt_file: Cleaned receptor PDBQT file to use for docking
            - center_pos: 3-dim list containing (x,y,z) coordinates of grid box
            - size: 3-dim list containing sizing information of grid box in (x,y,z) directions
            - n_conformers: how many times are we docking each SMILES string?
            - get_pose_str: Return output pose as string (True) or not (False)
            - timeout_duration: Timeout in seconds before new process automatically stops
            - additional_vina_args: Dictionary of additional Vina command arguments (e.g. {"cpu": "5"})
            - preparator: Function/Class callable to prepare molecule for docking. Should take the \
                argument format (smiles strings, ligand paths)
            - vina_cwd: Change current working directory of Vina shell (sometimes needed for GPU versions \
                and incorrect openCL pathing)
            - gpu_ids: GPU ids to use for multi-GPU docking (0 is default for single-GPU nodes). If None, \
                use all GPUs.
            - docking_attempts: Number of docking attempts to make on each GPU.
            - print_msgs: Show Python print messages in console (True) or not (False)
            - print_vina_output: Show Vina docking output in console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
        """

        if not os.path.isfile(receptor_pdbqt_file):
            raise Exception(rf"Receptor file: {receptor_pdbqt_file} not found")

        if len(center_pos) != 3:
            raise Exception(f"center_pos must contain 3 values: {center_pos} was provided")

        if len(size) != 3:
            raise Exception(f"size must contain 3 values: {size} was provided")

        # Getting all available GPU ids
        with subprocess.Popen(
            f"nvidia-smi --list-gpus", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        ) as proc:
            if proc.wait(timeout=timeout_duration) == 0:
                out = proc.stdout.read().decode("ascii")
                pattern = r"GPU (\d+):"
                available_gpu_ids = [int(x) for x in re.findall(pattern, out)]
            else:
                raise Exception(
                    f"Command 'nvidia-smi --list-gpus' returned unsuccessfully: {proc.stderr.read()}"
                )

        # Checking for incorrect GPU id input
        if gpu_ids is None:
            gpu_ids = available_gpu_ids
        elif type(gpu_ids) is int:
            if gpu_ids not in available_gpu_ids:
                raise Exception(f"Unknown GPU id: {gpu_ids}")
        else:
            unknown_gpu_ids = []
            for gpu_id in gpu_ids:
                if gpu_id not in available_gpu_ids:
                    unknown_gpu_ids.append(gpu_id)
            if len(unknown_gpu_ids) > 0:
                raise Exception(f"Unknown GPU id(s): {unknown_gpu_ids}")

        self.vina_cmd = vina_cmd
        self.receptor_pdbqt_file = os.path.abspath(receptor_pdbqt_file)
        self.center_pos = center_pos
        self.size = size
        self.n_conformers = n_conformers
        self.get_pose_str = get_pose_str
        self.timeout_duration = timeout_duration
        self.additional_vina_args = additional_vina_args
        self.preparator = preparator
        self.vina_cwd = vina_cwd
        self.gpu_ids = gpu_ids
        self.docking_attempts = docking_attempts
        self.print_msgs = print_msgs
        self.print_vina_output = print_vina_output
        self.debug = debug

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()

    def __call__(self, smi: Union[str, List[str]]) -> Optional[List[Union[float, None]]]:
        """
        Parameters:
        - smi: SMILES strings to perform docking. A single string activates single-ligand docking mode, while \
            multiple strings utilizes batched docking (if Vina version allows it).
        """
        if type(smi) is str:
            smi = [smi]

        for i in range(len(smi)):
            mol = Chem.MolFromSmiles(smi[i])
            if mol is not None:
                smi[i] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        if len(smi) > 0:
            return self._batched_docking(smi)
        else:
            return None

    def _batched_docking(self, smis: Iterable[str]) -> List[Union[float, None]]:
        # make temp pdbqt directories.
        ligand_tempdir = TemporaryDirectory(suffix="_lig")
        output_tempdir = TemporaryDirectory(suffix="_out")
        config_tempdir = TemporaryDirectory(suffix="_config")
        ligand_dir_path = ligand_tempdir.name
        output_dir_path = output_tempdir.name
        config_dir_path = config_tempdir.name

        # make different directories to support parallelization across multiple GPUs.
        for gpu_id in self.gpu_ids:
            make_dir(f"{ligand_dir_path}/{gpu_id}/", exist_ok=True)
            make_dir(f"{output_dir_path}/{gpu_id}/", exist_ok=True)

        # create hashed filename for each unique smiles
        ligand_path_fn = lambda smi: [
            os.path.abspath(f"{ligand_dir_path}/{sanitize_smi_name_for_file(smi)}_{i}.pdbqt")
            for i in range(self.n_conformers)
        ]

        # create hashed output filename for each unique smiles
        output_path_fn = lambda smi: [
            os.path.abspath(f"{output_dir_path}/{sanitize_smi_name_for_file(smi)}_{i}_out.pdbqt")
            for i in range(self.n_conformers)
        ]

        # not the fastest implementation, but safe if multiple experiments running at same time (with different tmp file paths)
        ligand_paths = []
        output_paths = []
        ligand_paths_by_smiles = []

        for i in range(len(smis)):
            current_ligand_paths = ligand_path_fn(smis[i])
            ligand_paths_by_smiles.append(current_ligand_paths)
            ligand_paths += current_ligand_paths
            output_paths += output_path_fn(smis[i])

        # Prepare ligands that don't have an existing output file (they aren't overlapping)
        if self.print_msgs:
            print("Preparing ligands...")
        self._prepare_ligands(smis, ligand_paths_by_smiles)

        # Multi-GPU docking: move ligands to the gpu_id directories
        split_ligand_paths = split_list(ligand_paths, len(self.gpu_ids))
        tmp_config_file_paths = []
        for i in range(len(self.gpu_ids)):
            gpu_id = self.gpu_ids[i]
            tmp_config_file_path = f"{config_dir_path}/config_{gpu_id}"
            tmp_config_file_paths.append(tmp_config_file_path)
            self._write_conf_file(
                tmp_config_file_path,
                {
                    "ligand_directory": f"{ligand_dir_path}/{gpu_id}/",
                    "output_directory": f"{output_dir_path}/{gpu_id}/",
                },
            )
            for ligand_file in split_ligand_paths[i]:
                try:
                    shutil.copy(ligand_file, os.path.abspath(f"{ligand_dir_path}/{gpu_id}/"))
                except FileNotFoundError:
                    if self.print_msgs:
                        print(f"Ligand file not found: {ligand_file}")

        # Perform docking procedure(s)
        if self.print_msgs:
            print("Ligands prepared. Docking...")

        vina_cmd_prefixes = [f"CUDA_VISIBLE_DEVICES={gpu_id} " for gpu_id in self.gpu_ids]

        # Run docking attempts multiple times on each GPU in case of failure.
        for attempt in range(self.docking_attempts):
            if self.debug:
                self.docking_profiler.time_it(
                    self._run_vina,
                    tmp_config_file_paths,
                    vina_cmd_prefixes=vina_cmd_prefixes,
                    blocking=False,
                )
            else:
                self._run_vina(
                    tmp_config_file_paths, vina_cmd_prefixes=vina_cmd_prefixes, blocking=False
                )
            if all(len(os.listdir(f"{output_dir_path}/{gpu_id}/")) > 0 for gpu_id in self.gpu_ids):
                break

            print(f"Docking attempt #{attempt + 1} failed on GPU {self.gpu_ids[i]}.")

        # Move files from temporary to proper directory (or delete if redoing calculation)
        for gpu_id in self.gpu_ids:
            move_files_from_dir(f"{ligand_dir_path}/{gpu_id}/", ligand_dir_path)

        for gpu_id in self.gpu_ids:
            move_files_from_dir(f"{output_dir_path}/{gpu_id}/", output_dir_path)

        # Something went horribly wrong
        if all(not os.path.exists(path) for path in output_paths):
            binding_scores = None

        else:
            # Gather binding scores
            binding_scores = []
            for i in range(len(output_paths)):
                binding_scores.append(self._get_output_score(output_paths[i]))

            if self.get_pose_str:
                binding_poses = []
                for i in range(len(output_paths)):
                    binding_poses.append(self._get_output_pose(output_paths[i]))

                binding_scores = (binding_scores, binding_poses)

        # clean up temp dirs
        ligand_tempdir.cleanup()
        output_tempdir.cleanup()
        config_tempdir.cleanup()
        if self.print_msgs:
            print("Docking complete.")
        return binding_scores

    def _prepare_ligands(
        self, smis: List[str], ligand_paths_by_smiles: List[List[str]]
    ) -> List[bool]:
        # Perform ligand preparation and save to proper path (tmp/non-tmp ligand dir)
        if self.debug:
            return self.preparation_profiler.time_it(self.preparator, smis, ligand_paths_by_smiles)
        else:
            return self.preparator(smis, ligand_paths_by_smiles)

    @staticmethod
    def _get_output_score(output_path: str) -> Union[float, None]:
        try:
            score = float("inf")
            with open(output_path, "r") as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", line)[0]
                        score = min(score, float(new_score))
            return score
        except FileNotFoundError:
            return None

    @staticmethod
    def _get_output_pose(output_path: str) -> Union[str, None]:
        try:
            with open(output_path, "r") as f:
                docked_pdbqt = f.read()
            return docked_pdbqt
        except FileNotFoundError:
            return None

    def _write_conf_file(self, config_file_path: str, args: Dict[str, str] = {}):
        conf = (
            f"receptor = {self.receptor_pdbqt_file}\n"
            + f"center_x = {self.center_pos[0]}\n"
            + f"center_y = {self.center_pos[1]}\n"
            + f"center_z = {self.center_pos[2]}\n"
            + f"size_x = {self.size[0]}\n"
            + f"size_y = {self.size[1]}\n"
            + f"size_z = {self.size[2]}\n"
        )

        for k, v in self.additional_vina_args.items():
            conf += f"{str(k)} = {str(v)}\n"

        for k, v in args.items():
            if (
                self.vina_cwd
            ):  # vina_cwd is not none, meaning we have to use global paths for config ligand and output dirs
                conf += f"{str(k)} = {os.path.join(os.getcwd(), str(v))}\n"
            else:
                conf += f"{str(k)} = {str(v)}\n"

        with open(config_file_path, "w") as f:
            f.write(conf)
        return conf

    def _run_vina(
        self,
        config_paths: List[List[str]],
        log_paths: List[List[str]] = None,
        vina_cmd_prefixes: List[str] = None,
        blocking: bool = True,
    ):
        """
        Runs Vina docking in separate shell process(es).
        """
        if log_paths is not None:
            assert len(config_paths) == len(log_paths)
        if vina_cmd_prefixes is not None:
            assert len(config_paths) == len(vina_cmd_prefixes)
        procs = []

        for i in range(len(config_paths)):
            if vina_cmd_prefixes is not None and vina_cmd_prefixes[i] is not None:
                cmd_str = vina_cmd_prefixes[i]
            else:
                cmd_str = ""

            cmd_str += f"{self.vina_cmd} --config {config_paths[i]}"

            if not self.print_vina_output:
                cmd_str += " > /dev/null 2>&1"

            proc = subprocess.Popen(cmd_str, shell=True, start_new_session=False, cwd=self.vina_cwd)

            if blocking:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                procs.append(proc)

        if not blocking:
            for proc in procs:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
