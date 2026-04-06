"""Chunked preprocessing for top dataset — avoids loading full file into memory."""
import h5py
import os
import numpy as np
import pandas as pd
import energyflow as ef
from optparse import OptionParser


def process_chunk(data, nparts=150):
    """Process a chunk of raw top data into (points, jets, pid)."""
    npid = data[:, -1]
    particles = data[:, :200 * 4].reshape((data.shape[0], -1, 4))

    jets = np.sum(particles, axis=1)
    jet_e = jets[:, 0]
    eta = ef.etas_from_p4s(jets)
    jets = ef.ptyphims_from_p4s(jets)
    jets[:, 1] = eta

    p_e = particles[:, :, 0]
    particles = ef.ptyphims_from_p4s(particles)
    particles[:, :, 3] = p_e
    mask = particles[:, :, 0] > 0

    jets = np.concatenate([jets, np.sum(mask, -1)[:, None]], -1)

    NFEAT = 7
    points = np.zeros((particles.shape[0], particles.shape[1], NFEAT))

    delta_phi = particles[:, :, 2] - jets[:, None, 2]
    delta_phi[delta_phi > np.pi] -= 2 * np.pi
    delta_phi[delta_phi <= -np.pi] += 2 * np.pi

    points[:, :, 0] = (particles[:, :, 1] - jets[:, None, 1])
    points[:, :, 1] = delta_phi
    points[:, :, 2] = np.ma.log(1.0 - particles[:, :, 0] / jets[:, None, 0]).filled(0)
    points[:, :, 3] = np.ma.log(particles[:, :, 0]).filled(0)
    points[:, :, 4] = np.ma.log(1.0 - particles[:, :, 3] / jet_e[:, None]).filled(0)
    points[:, :, 5] = np.ma.log(particles[:, :, 3]).filled(0)
    points[:, :, 6] = np.hypot(points[:, :, 0], points[:, :, 1])

    points *= mask[:, :, None]
    points = points[:, :nparts]

    jets = np.delete(jets, 2, axis=1)

    return points, jets, npid


if __name__ == '__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=150, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='top_dataset', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='train.h5', help="Input file name")
    parser.add_option("--chunk_size", type=int, default=20000, help="Events per chunk")

    (flags, args) = parser.parse_args()

    input_path = os.path.join(flags.folder, flags.sample)
    output_path = os.path.join(flags.folder, flags.sample.replace('.h5', '_ttbar.h5'))

    store = pd.HDFStore(input_path, 'r')
    nrows = store.get_storer('table').nrows
    store.close()
    print(f"Total rows: {nrows}, chunk size: {flags.chunk_size}")

    first = True
    offset = 0
    while offset < nrows:
        end = min(offset + flags.chunk_size, nrows)
        print(f"  Processing rows {offset}–{end}...", flush=True)

        store = pd.HDFStore(input_path, 'r')
        chunk = store.select('table', start=offset, stop=end).values
        store.close()

        points, jets, pid = process_chunk(chunk, flags.npoints)

        if first:
            with h5py.File(output_path, 'w') as fh5:
                fh5.create_dataset('data', data=points, maxshape=(None, points.shape[1], points.shape[2]))
                fh5.create_dataset('jet', data=jets, maxshape=(None, jets.shape[1]))
                fh5.create_dataset('pid', data=pid, maxshape=(None,))
            first = False
        else:
            with h5py.File(output_path, 'a') as fh5:
                for key, arr in [('data', points), ('jet', jets), ('pid', pid)]:
                    fh5[key].resize(fh5[key].shape[0] + arr.shape[0], axis=0)
                    fh5[key][-arr.shape[0]:] = arr

        offset = end

    with h5py.File(output_path, 'r') as fh5:
        print(f"Done: {output_path}")
        for k in fh5:
            print(f"  {k}: {fh5[k].shape}")
