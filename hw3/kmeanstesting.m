Sset = size(99);
Dset = size(99);
SSEset = size(1499);
SoverDset = size(99);
caseNormMean = csvread('normalizedCase.csv', 1, 1);
casecsv = caseNormMean';
for k = 2:2000
    k
    [idx,C,sumd]=kmeans(casecsv,k,'MaxIter',1000);
    
%     Dkm = pdist2(C,C);
%     Dk = sum(sum(Dkm),2);
%     [rd,cd] = size(Dkm);
%     cpNum = rd*cd;
%     D = Dk/cpNum;
%  Sk = 0;
%  for clusterIndex = 1:k
%         pointCollection = [];
%         for pointIndex = 1:8560
%             if idx(pointIndex) == clusterIndex
%                 point = casecsv(pointIndex);
%                 pointCollection = [pointCollection;point];
%             end
%         end
%         S1m=pdist2(pointCollection,pointCollection);
%         S1 = sum(sum(S1m),2);
%         [rs,cs] = size(S1m);
%         ppNum=rs*cs;
%         S1 = S1/ppNum;
%         Sk = Sk + S1;
%  end
%         S = Sk/k;
        
        

      Dk = 0;
      for i = 1:k
          for j = 1:k
                Dk = Dk + 1/(1+norm(C(i)-C(j)));
%               Dk = Dk + dot(C(i),C(j));
          end
      end
      D = Dk/(k*k);
      
      Sk = 0;
      SSE = 0;
      for clusterIndex = 1:k
         S1 = 0;
         SSE1 = 0;
         pointCollection = [];
         for pointIndex = 1:8560
             if idx(pointIndex) == clusterIndex
                 point = casecsv(pointIndex);
                 pointCollection = [pointCollection;point];
             end
         end
         [pNum,fNum] = size(pointCollection);
         for pointLeft = 1:pNum
             for pointRight = 1:pNum
                  S1 = S1 + 1/(1+norm(pointCollection(pointLeft)-pointCollection(pointRight)));
%                  S1 = S1 + dot(pointCollection(pointLeft),pointCollection(pointRight));
             end
         end
         Sk = Sk + S1/(pNum*pNum);
         
         for b = 1:pNum
             SSE = SSE + norm(pointCollection(b)-C(clusterIndex));
         end
      end
      
      S = Sk/k;




%     [idx,C,sumd]=kmeans(casecsv,k);
%     pairsNumD = 1/2*(k*k-k);
%     Dk = 0;
%     for left = 1:k-1
%         for right = left+1:k
%             X = [C(left);C(right)];
%            d = pdist(X,'cosine');
% %              d = dot(C(left),C(right));
%             Dk = Dk + d;
%         end
%     end
%     Sk = 0;
%     for clusterIndex = 1:k
%         oneClusterSim = 0;
%         pointCollection = [];
%         for pointIndex = 1:8560
%             if idx(pointIndex) == clusterIndex
%                 point = casecsv(pointIndex);
%                 pointCollection = [pointCollection;point];
%             end
%         end
%         [r,c] = size(pointCollection);
%         pointNum = r;
%         pairsNum = (pointNum*pointNum - pointNum)/2;
%         Ssum = 0;
%         for Left = 1:pointNum-1
%             for Right = Left+1:pointNum
%                  Y=[pointCollection(Left);pointCollection(Right)];
%                d = pdist(Y,'cosine');
% %                  d = dot(pointCollection(Left),pointCollection(Right));
%                 Ssum = Ssum + d;
%             end
%         end
%         if pairsNum == 0
%             oneCluster = 0;
%         else
%             oneCluster = Ssum/pairsNum;
%         end
%         Sk = Sk + oneCluster;
%     end
%     
%     S = Sk/k;
%     D = Dk/pairsNumD;
   
    Sset
    Dset
    SSEset(k-1) = SSE;
    Sset(k-1) = S; 
    Dset(k-1) = D;
    SoverDset(k-1) = S/D;
end