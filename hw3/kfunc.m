function [av_icc, av_ice, av_bcc, av_bce] = kfunc(X, k)
    idx = kmeans(X, k);
    idc = kmeans(X, k, 'Distance','cosine');
    sum(idx ~= idc);
    cluster_cos = [X, idc];
    cluster_data = [X, idx];
    ic = zeros(k, 1);
    %cluster_data = sortrows(cluster_data, size(cluster_data, 2));
%     in_cluster = zeros(size(X, 1), 1);
    in_cluster_euc = [];
    in_cluster_cos = [];
    for i = 1:k
        single_cluster_euc = [];
        single_cluster_cos = [];
        kclus = cluster_data(:,size(cluster_data, 2)) == i;
        kclus_cos = cluster_cos(:,size(cluster_data, 2)) == i;
        kclus_cosmat = cluster_data(kclus_cos,:);
        kclus_mat = cluster_data(kclus,:);
        % euclidean in cluster
        for j = 1:size(kclus_mat)
            for q = 1:size(kclus_mat)
                if j~=q
                    single_cluster_euc = [single_cluster_euc dot(kclus_mat(j,:), kclus_mat(q,:));];
                end
            end
        end
        in_cluster_euc = [in_cluster_euc mean(single_cluster_euc)];
        % cosine in cluster
        for j = 1:size(kclus_cosmat)
            for q = 1:size(kclus_cosmat)
                if j~=q
                    single_cluster_cos = [single_cluster_cos dot(kclus_cosmat(j,:), kclus_cosmat(q,:));];
                end
            end
        end
        in_cluster_cos = [in_cluster_cos mean(single_cluster_cos)];

    end
    av_icc = mean(in_cluster_cos);
    av_ice = mean(in_cluster_euc);
end

        
        
 
    
