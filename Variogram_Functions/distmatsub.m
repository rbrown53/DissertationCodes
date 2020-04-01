function iid = distmatsub(i)
    n=128^2;
    j  = (i+1:n)'; 
    d = sqrt(sum(bsxfun(@minus,X(i,:),X(j,:)).^2,2));
    I  = d<=dmax;
    iid = [j(I) d(I)];
end