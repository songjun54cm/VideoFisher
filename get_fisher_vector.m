run("fisher_params.m");

tic;
video_feats = load(all_feat_file_path, 'all_video_feat');
toc;
disp(['load video feature with shape : ', num2str(size(video_feats))]);

run([vlfeat_dir, 'toolbox/vl_setup']);

opt = struct();
opt.sample_number = sample_number;
opt.pca_number = pca_number;
opt.gmm_n = gmm_cluster_n;
opt.fisher_norm = fisher_norm;

## sampling feature
if sample_number >0
    sample_feats = video_feats(randsample(size(video_feats,1), sample_number), :, :, :);
else
    sample_feats = video_feats;
end
feat_size = size(sample_feats);
sample_feats = reshape(sample_feats, prod(feat_size(1:3)), feat_size(4));

## pca
if pca_number >0
    norm_f, mu, sigma] = zscore(sample_feats);
    pcaEigenVs = princomp(norm_f);
    pcaTransMatrix = pcaEigenVs(:,1:pca_number);
    save([save_folder, 'pcaTransMatrix.mat'], 'pcaTransMatrix');
    save([save_folder, 'pcaMu.mat'], 'mu');
    save([save_folder, 'pcaSigma.mat'], 'sigma');
    disp(['save pca values to ', save_folder]);

    opt.pca_mu = mu;
    opt.pca_sigma = sigma;
    opt.pca_transmatrix = pcaTransMatrix;

    sample_feats = norm_f * pcaTransMatrix;
end

[means, covars, priors] = vl_gmm(sample_feats',  gmm_cluster_n);
opt.means = means;
opt.covars = covars;
opt.priors = priors;
save([save_folder, 'gmm_means.mat'], 'means');
save([save_folder, 'gmm_covars.mat'], 'covars');
save([save_folder, 'gmm_priors.mat'], 'priors');

if pca_number>0
    n1 = pca_number;
else
    n1 = size(video_feats, 4);
end
encs = zeros(size(video_feats,1), size(means,2)*n1*2);

for i = 1:size(video_feats,1)
    vfs = reshape(video_feats(i,:,:,:), size(video_feats,2)*size(video_feats,3), size(video_feats,4));
    if opt.pca_number>0
        norm_vfs = (vfs - repmat(opt.pca_mu, size(vfs,1), 1)) ./ repmat(opt.pca_sigma, size(vfs, 1), 1);
        norm_vfs = norm_vfs * opt.pca_transmatrix;
    else
        norm_vfs = vfs;
    end
    if length(opt.fisher_norm)>=1
        v_enc = vl_fisher(norm_vfs', opt.means, opt.covars, opt.priors, opt.fisher_norm);
    else
        v_enc = vl_fisher(norm_vfs', opt.means, opt.covars, opt.priors);

    end
    encs(i,:) = v_enc';
end
disp(['size encode vectors: ', num2str(size(encs)), ', save to ', save_folder]);
save([save_folder, 'enc_features.mat'], 'encs');
