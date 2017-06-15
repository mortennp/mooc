function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);
    hidden_state_0_probs = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_state_0 = sample_bernoulli(hidden_state_0_probs);
    recon_probs = hidden_state_to_visible_probabilities(rbm_w, hidden_state_0);
    recon = sample_bernoulli(recon_probs);
    hidden_state_1_probs = visible_state_to_hidden_probabilities(rbm_w, recon);
    %hidden_state_1 = sample_bernoulli(hidden_state_1_probs);
    hidden_state_1 = hidden_state_1_probs;
    ret = (hidden_state_0 * transpose(visible_data) - hidden_state_1 * transpose(recon))/size(visible_data,2);
end
