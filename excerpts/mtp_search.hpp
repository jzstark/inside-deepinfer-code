update_best(); // All the golden tokens

for (auto &acc : acc_queue_for_bsz) {
    cur_expected += acc.first;
    cur_bsz += 1;
    update_best();
}
