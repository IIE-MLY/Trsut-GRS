import torch
import torch.nn.functional as F
import numpy as np

def weight_bpr_loss(user_emb, pos_item_emb, neg_item_emb, user_real):

    weight = [1 / (i + 10e-8) for i in user_real]
    pos_score_matrix = torch.mul(user_emb, pos_item_emb) * torch.tensor(np.array(weight)[:, np.newaxis]).cuda()
    pos_score = pos_score_matrix.sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def wrmf_loss(user_emb, pos_item_emb, neg_item_emb,pos_weight=20):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = pos_weight*((pos_score-1)**2)+(neg_score-0)**2
    return loss.sum()

def alignment_loss(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg

def l2_reg_loss_weight(reg, user_real, item_real, user_emb, item_emb):
    user_weight = [1 / (i + 10e-8) for i in user_real]
    item_weight = [1 / (i + 10e-8) for i in user_real]
    emb_loss = 0
    for i, user_emb in enumerate(user_emb):
        emb_loss += user_weight[i] * torch.norm(user_emb, p=2)
    for i, item_ in enumerate(item_emb):
        emb_loss += item_weight[i] * torch.norm(item_, p=2)
    return emb_loss * reg

def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) # 行求和
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def InfoNCE_weight(view1, view2, temperature, weight):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = ttl_score * weight
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) # 行求和
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p+kl_q)