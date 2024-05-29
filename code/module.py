import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils.protein.constants import BBHeavyAtom
from common_utils.modules.layers import LayerNorm, AngularEncoding
from common_utils.modules.geometry import global_to_local, local_to_global, construct_3d_basis, angstrom_to_nm, pairwise_dihedrals, per_directions, pairwise_directions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim, args, max_aa_types=22):
        super().__init__()
        self.args = args
        self.aatype_embed = nn.Embedding(max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.mut_embed = nn.Embedding(num_embeddings=2, embedding_dim=int(feat_dim/2), padding_idx=0)

        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(7) + 3 * 3 + int(feat_dim/2)

        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, batch, pretrain=False):

        N, L = batch['aa'].size() # (N, L)
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]

        # Amino acid identity features
        if pretrain:
            input_aa = copy.deepcopy(batch['aa'])
            mask_aa = (torch.bernoulli(torch.full(size=(N, L), fill_value=self.args.mask_ratio)).to(device) * batch['mask_atoms'][:, :, 1]).bool() # (N, L)
            aa_noise = torch.randint(0, 22, size=(N, L)).to(device) # (N, L)
            input_aa[mask_aa] = aa_noise[mask_aa]
        else:
            input_aa = batch['aa']

        aa_feat = self.aatype_embed(input_aa) # (N, L, F)

        # Dihedral features
        dihedral = torch.cat([batch['phi'][..., None], batch['psi'][..., None], batch['omega'][..., None], batch['chi']], dim=-1) # (N, L, 7)
        dihedral_mask = torch.cat([batch['phi_mask'][..., None], batch['psi_mask'][..., None], batch['omega_mask'][..., None], batch['chi_mask']], dim=-1) # (N, L, 7)
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 7, F)
        dihedral_feat = dihedral_feat.reshape(N, L, -1) # (N, L, 7*F)

        # Direction features
        direct_feat = per_directions(batch['pos_atoms']) # (N, L, F)

        if pretrain:
            input_dihedral_feat = copy.deepcopy(dihedral_feat)
            
            mask_angle = (torch.bernoulli(torch.full(size=(N, L), fill_value=self.args.mask_ratio)).to(device) * batch['mask_atoms'][:, :, 1]).bool() # (N, L)
            input_dihedral_feat[mask_angle] *= 0
        else:
            input_dihedral_feat = dihedral_feat

        # Mutation features
        mut_feat = self.mut_embed(batch['mut_flag'].long()) # (N, L, F)

        # Node features
        out_feat = self.out_mlp(torch.cat([aa_feat, input_dihedral_feat, direct_feat, mut_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]

        if pretrain:
            return out_feat, [dihedral_feat, mask_aa, mask_angle]
        else:
            return out_feat
    

class ResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms=5, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos

        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(3)

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim + 3 * 4

        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, batch):

        N, L = batch['aa'].size()
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :] # (N, L, L)

        # Pair identity features
        aa_pair = batch['aa'][:,:,None] * self.max_aa_types + batch['aa'][:,None,:] # (N, L, L)
        aa_pair_feat = self.aa_pair_embed(aa_pair) # (N, L, L, F)
    
        # Relative position features
        same_chain = (batch['chain_nb'][:, :, None] == batch['chain_nb'][:, None, :]) # (N, L, L)
        relpos = torch.clamp(batch['res_nb'][:,:,None] - batch['res_nb'][:,None,:], min=-self.max_relpos, max=self.max_relpos) # (N, L, L)
        relpos_feat = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # Distance features
        d = angstrom_to_nm(torch.linalg.norm(batch['pos_atoms'][:,:,None,:,None] - batch['pos_atoms'][:,None,:,None,:], dim=-1, ord=2)).reshape(N, L, L, -1) # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair)) # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (batch['mask_atoms'][:,:,None,:,None] * batch['mask_atoms'][:,None,:,None,:]).reshape(N, L, L, -1)
        dist_feat = self.distance_embed(d_gauss * mask_atom_pair)

        # Orientation features
        dihed = pairwise_dihedrals(batch['pos_atoms']) # (N, L, L, 3)
        dihed_feat = self.dihedral_embed(dihed)

        # Direction features
        direct_feat = pairwise_directions(batch['pos_atoms'])

        # Edge features
        feat_all = self.out_mlp(torch.cat([aa_pair_feat, relpos_feat, dist_feat, dihed_feat, direct_feat], dim=-1)) # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None] # (N, L, L, F)

        return feat_all
    

def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L, L, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    mask_pair = mask.unsqueeze(-1).expand_as(logits)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, H)

    alpha = torch.where(mask_pair, alpha, torch.zeros_like(alpha))

    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, hidden_dim=32, num_points=8, num_heads=8, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)

        # Output
        self.mlp_transition_1 = nn.Linear((num_heads * pair_feat_dim) + (num_heads * hidden_dim) + (num_heads * num_points), node_feat_dim)
        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition_2 = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                              nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                              nn.Linear(node_feat_dim, node_feat_dim * codebook_head))
        self.layer_norm_2 = LayerNorm(node_feat_dim * codebook_head)

    def attention_logits(self, R, t, x, z):
        N, L, _ = t.size()

        query_node = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        key_node = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        logits_node = (query_node.unsqueeze(2) * key_node.unsqueeze(1) * (1 / np.sqrt(self.hidden_dim))).sum(-1)  # (N, L, L, H)

        logits_pair = self.proj_pair(z)
    
        query_point = _heads(self.proj_query_point(x), self.num_heads * self.num_points, 3)  # (N, L, H*P 3)
        query_point = local_to_global(R, t, query_point)  # (N, L, H*P 3)
        query_point = query_point.reshape(N, L, self.num_heads, -1)  # (N, L, H, P*3)

        key_point = _heads(self.proj_key_point(x), self.num_heads * self.num_points, 3)  # (N, L, H*P 3)
        key_point = local_to_global(R, t, key_point)  # (N, L, H*P 3)
        key_point = key_point.reshape(N, L, self.num_heads, -1)  # (N, L, H, P*3)

        sum_sq_dist = ((query_point.unsqueeze(2) - key_point.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, H)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_points))) / 2)  # (N, L, L, H)

        return (logits_node + logits_pair + logits_spatial) * np.sqrt(1 / 3)
    
    def node_aggregation(self, alpha, R, t, x, z):
        N, L = x.shape[:2]

        value_node = _heads(self.proj_value(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        feat_node = alpha.unsqueeze(-1) * value_node.unsqueeze(1)  # (N, L, L, H, *) @ (N, *, L, H, F)
        feat_node = feat_node.sum(dim=2).reshape(N, L, -1)  # (N, L, H*F)

        feat_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, H, *) @ (N, L, L, *, F)
        feat_pair = feat_pair.sum(dim=2).reshape(N, L, -1)  # (N, L, H*F)

        value_point = _heads(self.proj_value_point(x), self.num_heads * self.num_points, 3)  # (N, L, H*P 3)
        value_point = local_to_global(R, t, value_point)  # (N, L, H*P 3)
        value_point = value_point.reshape(N, L, self.num_heads, self.num_points, -1)  # (N, L, H, P, 3)

        feat_spatial = alpha.unsqueeze(-1).unsqueeze(-1) * value_point.unsqueeze(1)  # (N, L, L, H, *, *) @ (N, *, L, H, P, 3)
        feat_spatial = feat_spatial.sum(dim=2)  # (N, L, H, P, 3)
        feat_spatial = global_to_local(R, t, feat_spatial)  # (N, L, H, P, 3)
        feat_spatial = feat_spatial.norm(dim=-1).reshape(N, L, -1)  # (N, L, H*P)

        return torch.cat([feat_node, feat_pair, feat_spatial], dim=-1)

    def forward(self, R, t, x, z, intra_mask, inter_mask):
        # Attention weights
        att_logits = self.attention_logits(R, t, x, z)
        att_ratio = (intra_mask.sum(-1).sum(-1) / (intra_mask.sum(-1).sum(-1) + inter_mask.sum(-1).sum(-1))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha = _alpha_from_logits(att_logits, intra_mask) * att_ratio + _alpha_from_logits(att_logits, inter_mask) * (1 - att_ratio)

        # Aggregate features
        agg_feats = self.node_aggregation(alpha, R, t, x, z)

        # Update features
        feats = self.layer_norm_1(x + self.mlp_transition_1(agg_feats))
        out_feats = self.mlp_transition_2(feats)

        if feats.shape[-1] == out_feats.shape[-1]:
            feats = self.layer_norm_2(feats + out_feats)
        else:
            feats = self.layer_norm_2(out_feats)

        return feats


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.blocks.append(GABlock(node_feat_dim, pair_feat_dim, codebook_head))
            else:
                self.blocks.append(GABlock(node_feat_dim, pair_feat_dim))
        self.args = args

    def edge_construction(self, batch, cutoff=15.0, K=15):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :] # (N, L, L)

        same_chain = (batch['chain_nb'][:, :, None] == batch['chain_nb'][:, None, :])
        diff_chain = (batch['chain_nb'][:, :, None] != batch['chain_nb'][:, None, :])

        seq_dis = torch.abs(batch['res_nb'][:,:,None] - batch['res_nb'][:,None,:])
        seq_mask = (seq_dis == torch.tensor(1)) + (seq_dis == torch.tensor(2))

        pairwise_dis = torch.linalg.norm(batch['pos_atoms'][:,:,None,BBHeavyAtom.CA] - batch['pos_atoms'][:,None,:,BBHeavyAtom.CA], dim=-1, ord=2)
        rball_mask = pairwise_dis < cutoff

        dis, _ = pairwise_dis.topk(K+1, dim=-1, largest=False)
        knn_mask = pairwise_dis < dis[:, :, K:K+1]

        intra_mask = ((rball_mask + knn_mask + seq_mask) * same_chain * mask_pair).bool()
        inter_mask = ((rball_mask + knn_mask) * diff_chain * mask_pair).bool()

        return intra_mask, inter_mask
    
    def forward(self, batch, res_feat, pair_feat):
        pos_atoms = batch['pos_atoms']
        R = construct_3d_basis(pos_atoms[:, :, BBHeavyAtom.CA], pos_atoms[:, :, BBHeavyAtom.C], pos_atoms[:, :, BBHeavyAtom.N])
        t = angstrom_to_nm(pos_atoms[:, :, BBHeavyAtom.CA])

        intra_mask, inter_mask = self.edge_construction(batch, cutoff=self.args.cutoff, K=self.args.knn)

        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, intra_mask, inter_mask)

        return res_feat


class SABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, n_channel=4, edge_type=2, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.coord_mlp = nn.ModuleList()
        self.relation_mlp = nn.ModuleList()

        self.message_mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2 + n_channel**2 + pair_feat_dim, node_feat_dim),
            nn.SiLU(),
            nn.Linear(node_feat_dim, node_feat_dim))

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 3, node_feat_dim),
            nn.SiLU(),
            nn.Linear(node_feat_dim, node_feat_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + pair_feat_dim + node_feat_dim, node_feat_dim),
            nn.SiLU(),
            nn.Linear(node_feat_dim, pair_feat_dim))
        
         
        for _ in range(edge_type):
            self.relation_mlp.append(nn.Linear(node_feat_dim, node_feat_dim, bias=False))

            layer = nn.Linear(node_feat_dim, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            self.coord_mlp.append(nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim),
                nn.SiLU(),
                layer
            ))

        self.layer_norm = LayerNorm(node_feat_dim)

    def coord2radial(self, coord):
        coord_diff = coord[:, :, None] - coord[:, None, :]  # [N, L, L, A, 3]
        radial = torch.einsum('ijkmn,ijknp->ijkmp', coord_diff, coord_diff.transpose(-1, -2))  # [N, L, L, A, A]
        radial = F.normalize(radial.reshape(-1, coord.shape[-2], coord.shape[-2]), dim=0).reshape(radial.shape)  # [N, L, L, A, A]

        return radial, coord_diff
    
    def message_updating(self, h, z, radial):
        N, L, _ = h.size()
        radial = radial.reshape(N, L, L, -1)  # [N, L, L, A * A]

        out = torch.cat([h[:,:,None].expand(-1, -1, L, -1), h[:,None,:].expand(-1, L, -1, -1), radial, z], dim=-1)
        out = self.message_mlp(out)  # [N, L, L, F]
        out = self.dropout(out)

        return out
    
    def node_updating(self, h, m, intra_mask, inter_mask):
        agg_intra = self.relation_mlp[0]((m * intra_mask[:,:,:,None]).sum(dim=-2))  # [N, L, F]
        agg_inter = self.relation_mlp[1]((m * inter_mask[:,:,:,None]).sum(dim=-2))  # [N, L, F]
            
        agg = torch.cat([h, agg_intra, agg_inter], dim=-1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = self.layer_norm(h + out)

        return out

    def edge_updating(self, h, z):
        _, L, _ = h.size()

        out = torch.cat([h[:,:,None].expand(-1, -1, L, -1), h[:,None,:].expand(-1, L, -1, -1), z], dim=-1)
        out = self.edge_mlp(out)

        return out

    def coord_updating(self, coord, coord_diff, m, intra_mask, inter_mask, R, t):
        agg_intra = coord_diff * self.coord_mlp[0](m).unsqueeze(-1) # [N, L, L, A, 3]
        agg_inter = coord_diff * self.coord_mlp[1](m).unsqueeze(-1) # [N, L, L, A, 3]
        agg = (agg_intra * intra_mask.unsqueeze(-1).unsqueeze(-1) + agg_inter * inter_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=2) # [N, L, A, 3]
        agg = agg / (intra_mask + inter_mask + 1e-6).sum(dim=2).unsqueeze(-1).unsqueeze(-1) # [N, L, A, 3]
        
        coord = coord + agg

        return coord
    
    def forward(self, R, t, coord, h, z, intra_mask, inter_mask):

        radial, coord_diff = self.coord2radial(coord)
        m = self.message_updating(h, z, radial)

        h = self.node_updating(h, m, intra_mask, inter_mask)
        z = self.edge_updating(h, z)
        coord = self.coord_updating(coord, coord_diff, m, intra_mask, inter_mask, R, t)

        return coord, h, z


class GADecoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args):
        super(GADecoder, self).__init__()
        self.blocks = nn.ModuleList([
            SABlock(node_feat_dim, pair_feat_dim) 
            for _ in range(num_layers)
        ])
        self.args = args
        self.noise_sigma = 1.0

    def edge_construction(self, batch, cutoff=15.0, K=15):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :] # (N, L, L)

        same_chain = (batch['chain_nb'][:, :, None] == batch['chain_nb'][:, None, :])
        diff_chain = (batch['chain_nb'][:, :, None] != batch['chain_nb'][:, None, :])

        seq_dis = torch.abs(batch['res_nb'][:,:,None] - batch['res_nb'][:,None,:])
        seq_mask = (seq_dis == torch.tensor(1)) + (seq_dis == torch.tensor(2))

        pairwise_dis = torch.linalg.norm(batch['pos_atoms'][:,:,None,BBHeavyAtom.CA] - batch['pos_atoms'][:,None,:,BBHeavyAtom.CA], dim=-1, ord=2)
        rball_mask = pairwise_dis < cutoff

        dis, _ = pairwise_dis.topk(K+1, dim=-1, largest=False)
        knn_mask = pairwise_dis < dis[:, :, K:K+1]

        intra_mask = ((rball_mask + knn_mask + seq_mask) * same_chain * mask_pair).bool()
        inter_mask = ((rball_mask + knn_mask) * diff_chain * mask_pair).bool()

        return intra_mask, inter_mask

    def forward(self, batch, res_feat, pair_feat):

        intra_mask, inter_mask = self.edge_construction(batch, cutoff=self.args.cutoff, K=self.args.knn)

        coord = copy.deepcopy(batch['pos_atoms'][:, :, :4])
        mask = (torch.bernoulli(torch.full(size=(coord.shape[0], coord.shape[1]), fill_value=self.args.mask_ratio)).to(device) * batch['mask_atoms'][:, :, 1]).bool() # (N, L)

        coord_noise = torch.normal(0, self.noise_sigma, size=(coord.shape[0], coord.shape[1], 1, coord.shape[3])).expand_as(coord).to(device) # (N, L, 4, 3)
        coord[mask] += coord_noise[mask]

        pos_atoms = batch['pos_atoms']
        R = construct_3d_basis(pos_atoms[:, :, BBHeavyAtom.CA], pos_atoms[:, :, BBHeavyAtom.C], pos_atoms[:, :, BBHeavyAtom.N])
        t = angstrom_to_nm(pos_atoms[:, :, BBHeavyAtom.CA])

        for block in self.blocks:
            coord, res_feat, pair_feat = block(R, t, coord, res_feat, pair_feat, intra_mask, inter_mask)

        return coord, mask