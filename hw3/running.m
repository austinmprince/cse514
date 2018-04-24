        %https://www.mathworks.com/matlabcentral/answers/36281-newbie-euclidean-distance-of-a-matrix
       % myEdist = squeeze(sqrt(sum(bsxfun(@minus, kclus_mat,reshape( kclus_mat',1,size( kclus_mat,2),size( kclus_mat,1))).^2,2))) %Engine
        
%         for i = 1:size(kclus_mat,1)
%             for j = 1:size(kclus_mat, 1)
%                 if i ~= j
%                     
%                 end
%             end
%                     
%         end

av_ic = zeros(98,1);
for i = 2:100
     [av_icc, av_ice, ] = kfunc(data, i)
     av_ic(i) = av_icc;
end
av_ic