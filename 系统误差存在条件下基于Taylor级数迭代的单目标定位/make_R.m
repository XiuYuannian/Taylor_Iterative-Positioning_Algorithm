function R = make_R(M, sig_tdoa, sig_groa, sig_aoa)
nd = M-1;
nz = nd + nd + M + M;
R = zeros(nz,nz);
R(1:M, 1:M) = (sig_aoa^2) * eye(M);
R(M+(1:M), M+(1:M)) = (sig_aoa^2) * eye(M);
R(2*M+(1:nd), 2*M+(1:nd)) = (sig_tdoa^2) * eye(nd);
R(2*M+nd+(1:nd), 2*M+nd+(1:nd)) = (sig_groa^2) * eye(nd);
end