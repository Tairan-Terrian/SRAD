from option import args
import pandas as pd
import numpy as np
import loader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)

def reindex(df, bipartite=False):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df

def dynamicStructRep(G, n, alpha, iter=100):
    nc = len(G)
    emb1 = 1 / n * np.ones((nc, n))
    emb = []
    for i in range(nc):
        emb.append(temporalAgg(emb1, G, i, alpha))
    emb = np.vstack(emb)
    count = getnumber(emb)
    for i in range(iter):
        scaler = MinMaxScaler()
        emb1 = scaler.fit_transform(emb)
        kmeans = KMeans(n_clusters=n, random_state=1).fit(emb1)
        val = kmeans.transform(emb1)
        M = val.max(axis=1)
        m = val.min(axis=1)
        subx = (M.reshape(nc, 1) - val) / (M - m).reshape(nc, 1)
        su = subx.sum(axis=1)
        subx = subx / su.reshape(nc, 1)
        emb2 = []
        for j in range(nc):
            emb2.append(temporalAgg(subx, G, j, alpha))
        emb = np.vstack(emb2)
        count1 = getnumber(emb)
        if count >= count1:
            break
        else:
            count = count1
    return emb


# Temporal aggregation
def temporalAgg(emb, G, i, alpha):
    s = emb.shape[1]
    M = np.zeros((s, s))
    a = []
    z = np.zeros((1, s))
    e = []
    r = np.zeros((1, s))
    for t in range(len(G[i])):
        (time, neighbors) = G[i][t]
        rep = aggregate(emb, G, i, neighbors, time)
        e.append(rep.reshape((s, 1)))
        r += rep
    e = np.array(e)
    for t in range(1, len(G[i])):
        (ptime, pneighbors) = G[i][t]
        (ltime, lneighbors) = G[i][t - 1]
        z = np.exp((ptime - ltime) / alpha) * (e[t - 1].transpose() + z)
        a = e[t] * z
        M += a

    M = M.flatten()
    M = np.hstack([M, r[0]])
    return M


# Neighbor Aggregation
def aggregate(emb, G, i, neighbors, time):
    rep = np.zeros((emb.shape[1],))
    for neigh in neighbors:
        rep += emb[neigh]
    return rep


# returns number of unique node representations for convergence calculation
def getnumber(emb):
    ss=set()
    for x in range(emb.shape[0]):
        sd=''
        for y in range(emb.shape[1]):
            sd+=','+str(emb[x,y])
        ss.add(sd)
    return len(ss)

def struct_representation(STRUCT_REP_PATH, args):
    origin_emb_path = '{}/or_{}_emb.csv'.format(args.dir_data, args.data_set)
    data = pd.read_csv(STRUCT_REP_PATH)
    l = loader.loader()
    l.read(data)
    emb = dynamicStructRep(l.G, args.clusters, args.alpha)
    l.storeEmb(origin_emb_path, emb)
    l.sortFeature(origin_emb_path, args)

if __name__=='__main__':
    dateset_dir = '{}/{}.csv'.format(args.dir_data, args.data_set)
    OUT_DF = '{}/ml_{}.csv'.format(args.dir_data, args.data_set)
    OUT_FEAT = '{}/ml_{}.npy'.format(args.dir_data, args.data_set)
    OUT_NODE_FEAT = '{}/ml_{}_node.npy'.format(args.dir_data, args.data_set)
    OUT_STRUCT_REP = '{}/struct_{}.csv'.format(args.dir_data, args.data_set)
    IN_FINAL_EMB_PATH = '{}/fi_{}_emb.csv'.format(args.dir_data, args.data_set)

    df, feat = preprocess(dateset_dir)
    new_df = reindex(df, args.bipartite)
    struct_df = new_df.iloc[:, :3].copy()
    struct_df.to_csv(OUT_STRUCT_REP, index=False)

    struct_representation(OUT_STRUCT_REP, args)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    data = pd.read_csv(IN_FINAL_EMB_PATH, header=None)
    rand_feat = data.iloc[:, 1:].to_numpy()
    new_row = np.zeros((1, rand_feat.shape[1]))
    rand_feat = np.vstack([new_row, rand_feat])


    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
