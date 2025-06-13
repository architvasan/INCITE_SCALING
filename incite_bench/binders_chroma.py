"""
# Setup
"""
import argparse
import antibody_design.utils.cif2pdb as cif2pdb
from collections import defaultdict, namedtuple
from chroma import Chroma, Protein, conditioners, api
from chroma.constants.sequence import AA20_3_TO_1
from dataclasses import dataclass
import os
import random
import sys
import torch

def vector_at_45_degrees(r):
    theta = torch.tensor(45.0) * torch.pi / 180  # Convert to radians
    phi = torch.tensor(45.0) * torch.pi / 180    # Convert to radians

    x = r * torch.cos(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.cos(phi)
    z = r * torch.sin(phi)

    return torch.tensor([x, y, z])


def sample_from_sphere(num_points, radius, center, device='xpu'):
    """Sample points uniformly from a sphere centered at 'center'."""
    # Sample from a normal distribution
    points = torch.randn(num_points, 3)
    # Normalize to make them lie on a sphere
    points = points / torch.norm(points, dim=1, keepdim=True)
    # Scale to the desired radius
    points = points * radius
    # Translate to the desired center
    print(points.shape)
    print(center.unsqueeze(0).shape)
    return points.to(device) + center.unsqueeze(0).to(device)


def catch(func, handle=lambda e : e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return None

@dataclass
class ChromaBinders:
    reg_key: str = "794e2a0f3aca44bf9ad8108cd58894b9"
    device: str = "xpu"

    input_dir: str = "./inputpdbs/"
    output_dir: str = "./output_nmnat2/"
    
    inp_pdbs: str| list[str] = 'meta0_cleaned.pdb'
    len_binder: int | list[int] = 30
    num_cycles: int = 5
    num_backbones: int = 10
    num_designs: int = 10
    diff_steps: int = 100
    
    hot_sphere: bool = True
    hotspot_indices: list[int] | None = None
    bind_sph_rad: int | None = None
    vector_displacement: float = 2.0
    weights_backbone: str = 'chroma_weights/90e339502ae6b372797414167ce5a632/weights.pt'
    weights_design: str = 'chroma_weights/03a3a9af343ae74998768a2711c8b7ce/weights.pt'
    weights_conditioner: str = './chroma_weights/3262b44702040b1dcfccd71ebbcf451d/weights.pt'
    centered_pdb_file: str = '2g3n'

    def __post_init__(self):
        """
        register api key
        """
        api.register_key(self.reg_key)

        """
        define chroma object
        """
        self.chroma = Chroma(
                        weights_backbone = self.weights_backbone,
                        weights_design = self.weights_design,
                        centered_pdb_file = self.centered_pdb_file
                        )

        """
        make output directory
        """
        os.makedirs(self.output_dir, exist_ok = True)

        if self.hot_sphere:
            assert self.hotspot_indices is not None
            self.hotspot_vector()

        """
        chroma aminoacid --> num and num --> aacid dicts
        """
        self.aa_to_num = {}
        self.num_to_aa = {}
        for i, aaa in enumerate(list(AA20_3_TO_1)):
            self.aa_to_num[AA20_3_TO_1[aaa]] = i
            self.num_to_aa[i] = AA20_3_TO_1[aaa]
    
    def setup_bind_hot(self, 
                       X_inp: object,
                       len_binder_rand: int | None,
                       ):
        if isinstance(self.len_binder, list):
            len_binder = len_binder_rand
        else:
            len_binder = self.len_binder

        num_points = len_binder * 4
        hotspot_indices_torch = torch.tensor(self.hotspot_indices)
        hotspot_coords = X_inp[:, hotspot_indices_torch, :]
        protein_cent = X_inp.mean(dim=(0, 1, 2)).to('xpu')
        hotspot_cent = hotspot_coords.mean(dim=(0, 1, 2)).to('xpu')  # Shape: (B, num_hotspots, 3)
        placement_vector = hotspot_cent - protein_cent
        unit_vector = placement_vector / torch.norm(placement_vector, p=2)
        #bind_sphere_center = hotspot_cent +\
        #                        vector_at_45_degrees(self.bind_sph_rad).to('xpu') +\
        #                        torch.tensor([0.5, 0.5, 0.5]).to('xpu')
        
        binder_coords = sample_from_sphere(num_points, self.bind_sph_rad, hotspot_cent).to('xpu')
        binder_coords += unit_vector * self.vector_displacement 
        binder_X = binder_coords.view(1, len_binder, 4, 3)
        return binder_X

    def prot_setup(self,
                   pdb_it: str,
                    ):

        # input pdb should be clean before, in this case only the chain A of pdb 6WRW
        protein = Protein(f"{self.input_dir}/{pdb_it}", device=self.device)
        X, C, S = protein.to_XCS()

        if isinstance(self.len_binder, list):
            len_binder = random.choice(self.len_binder)
        else:
            len_binder = self.len_binder
        
        if self.hot_sphere == True:
            binder_X = self.setup_bind_hot(
                                    X,
                                    len_binder
                                    )
            X_new = torch.cat(
                          [X,
                          binder_X.to(self.device),
                          ],
                          dim=1
                          )
        else:
            X_new = torch.cat(
                         [X,
                         torch.zeros(1, len_binder, 4, 3).to(self.device)
                         ],
                         dim=1
                         )

        C_new = torch.cat(
            [C,
             torch.full((1, len_binder), 2).to(self.device)
            ],
            dim=1
        )
        
        S_new = torch.cat(
            [S,
             torch.full((1, len_binder), 0).to(self.device)
            ],
            dim=1
        )
        
        del X,C,S

        protein = Protein(X_new, C_new, S_new, device=self.device)
        X, C, S = protein.to_XCS()

        return protein, X, C, S

    def conditioners_mask(self,
                     C: object,
                     S: object,
                     protein: object
                     ):
        L_binder = (C == 2).sum().item()
        L_receptor = (C == 1).sum().item()
        L_complex = L_binder+L_receptor
        assert L_complex==C.shape[-1]
        
        # keep original seqs of receptor by providing the mask
        mask_aa = torch.Tensor(L_complex * [[0] * 20])
        for i in range(L_complex):
            if i not in range(L_receptor):
                mask_aa[i] = torch.Tensor([1] * 20)
                mask_aa[i][S[0][i].item()] = 0
        
        mask_aa = mask_aa[None].to('xpu')#cuda()
        
        residues_to_keep = [i for i in range(L_receptor)]
        protein.sys.save_selection(gti=residues_to_keep, selname="receptor")
        conditioner_struc_R = conditioners.SubstructureConditioner(
                protein,
                backbone_model=self.chroma.backbone_network,
                selection = 'namesel receptor').to(self.device)
        
        conditioner_beta = conditioners.ProClassConditioner('cath', "2", weight=5, max_norm=20, model = self.weights_conditioner ).to(self.device)
        
        conditioner = conditioners.ComposedConditioner([conditioner_struc_R, conditioner_beta, ])
        
        return mask_aa, conditioner, protein


    def chroma_sampling(self,
                   protein,
                   conditioner,
                   mask_aa
                    ):
           proteins, traj = self.chroma.sample(
               protein_init=protein,
               conditioner=conditioner,
               design_selection = mask_aa,
               langevin_factor=2,
               langevin_isothermal=True,
               inverse_temperature=8.0,
               sde_func='langevin',
               full_output=True,
               steps=self.diff_steps,
               samples=self.num_backbones,
               num_designs=self.num_designs,
           )
           
           return proteins, traj

    def write_prot_cif(self,
                       proteins: list[object] | object,
                       out_phrase: str,
                        ):
           for it, prot in enumerate(proteins):
               prot.to(
                    f"{self.output_dir}/{out_phrase}_{it}.cif"
                    )
               cif2pdb.cif2pdb(f"{self.output_dir}/{out_phrase}_{it}.cif")

           del(proteins)

    def run(self):
        if isinstance(self.inp_pdbs, list):
            for pdb_it in self.inp_pdbs:
                protein, X, C, S = self.prot_setup(pdb_it)

                mask_aa,\
                conditioner,\
                protein = self.conditioners_mask(
                            C,
                            S,
                            protein)

                for loop in range(self.num_cycles):
                    proteins, trajs = self.chroma_sampling(
                                        protein,
                                        conditioner,
                                        mask_aa
                                         )

                    if self.hot_sphere == True:
                        self.write_prot_cif(
                                proteins,
                                f"{os.path.splitext(pdb_it)[0]}_{self.hotspot_indices[0]}_{loop}")

                    else:
                        self.write_prot_cif(
                            proteins,
                            f"{os.path.splitext(pdb_it)[0]}_{loop}"
                            )
        else:
            protein, X, C, S = self.prot_setup(self.inp_pdbs)

            mask_aa,\
            conditioner,\
            protein = self.conditioners_mask(
                        C,
                        S,
                        protein)

            for loop in range(self.num_cycles):
                proteins, trajs = self.chroma_sampling(
                                    protein,
                                    conditioner,
                                    mask_aa
                                     )

                self.write_prot_cif(
                        proteins,
                        f"{os.path.splitext(pdb_it)[0]}_{loop}")




def main():
    parser = argparse.ArgumentParser(description="Inputs for chroma binder design")

    reg_key: str = "794e2a0f3aca44bf9ad8108cd58894b9"
    device: str = "xpu"

    parser.add_argument(
            "-i", 
            "--input_dir",
            type=str,
            required=True,
            help="Input directory for pdbs"
            )

    parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            required=True,
            help="output directory for diffused pdbs"
            )

    parser.add_argument(
            "-ip",
            "--inp_pdbs",
            nargs="+",
            help="input pdbs (not full path, just base), list[str] or str"
            )

    parser.add_argument(
            "-l",
            "--len_binder",
            nargs="+",
            type=int,
            help="length of binders, list[int] or int")

    parser.add_argument(
            "-C",
            "--num_cycles",
            type=int,
            default=10,
            required=False,
            help="number of cycles")

    parser.add_argument(
            "-nb",
            "--num_backbones",
            type=int,
            default=10,
            required=False,
            help="number of backbones")

    parser.add_argument(
            "-nd",
            "--num_designs",
            type=int,
            default=10,
            required=False,
            help="number of designs")

    parser.add_argument(
            "-ds",
            "--diff_steps",
            type=int,
            default=100,
            required=False,
            help="number of diffusion steps")

    parser.add_argument(
            "-hs",
            "--hot_sphere",
            action="store_true",
            help="should we use hotspotting? default = False")

    parser.add_argument(
            "-hi",
            "--hotspot_indices",
            nargs="+",
            type=int,
            required=False,
            default=None)

    parser.add_argument(
            "-bsr",
            "--bind_sph_rad",
            type=int,
            required=False,
            default=None)

    args = parser.parse_args()
    reg_key: str = "794e2a0f3aca44bf9ad8108cd58894b9"
    device: str = "xpu"

    chroma_binder_obj = ChromaBinders(
                            reg_key = reg_key,
                            device = device,
                            input_dir = args.input_dir,
                            output_dir = args.output_dir,
                            inp_pdbs = args.inp_pdbs,
                            len_binder = args.len_binder,
                            num_cycles = args.num_cycles,
                            num_backbones = args.num_backbones,
                            num_designs = args.num_designs,
                            diff_steps = args.diff_steps,
                            hot_sphere = args.hot_sphere,
                            hotspot_indices = args.hotspot_indices,
                            bind_sph_rad = args.bind_sph_rad,
                            )
    
    chroma_binder_obj.run()


if __name__ == "__main__":

    main() 

