import NearestNeighbors
import Posegraph

function icp(
    surfels1::T,
    surfels2::T,
    T21_init::Matrix{<:AbstractFloat};
    iteration_num::Int64=10,
    max_nn_dist::Float64=0.3
) where T<:Union{GlSurfelsData, SurfelsData}

    xyz1 = get_xyz(surfels1)
    xyz2 = get_xyz(surfels2)

    T21 = copy(T21_init)

    ## build KD tree for search

    # Create trees
    kdtree = NearestNeighbors.KDTree(xyz2; leafsize = 10)
    for iter_n=1:iteration_num
        xyz1_trans = CameraModel.applyT(T21, xyz1)
        idxs, dists = NearestNeighbors.knn(kdtree, xyz1_trans, 1);

        dists2 = map(x->x[1], dists)

        g = Posegraph.Graph()

        v_T21 = Posegraph.VertexSE3("T21", T21)
        Posegraph.add_vertex!(g, v_T21)

        for (idx1, (idx2, dist)) = enumerate(zip(idxs, dists))
            if dist[1] < max_nn_dist
                pt1 = xyz1[:, idx1]
                pt2 = xyz2[:, idx2]
                e = Posegraph.Point3DistanceError(v_T21, pt1, pt2)
                Posegraph.add_edge!(g, e)
            end
        end

        Posegraph.solve!(g)
        T21 .= Posegraph.getT(v_T21)

    end
    return T21
end
