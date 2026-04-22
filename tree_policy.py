# Auto-generated decision tree policy
# Feature keys: ['top1_prob', 'target_top1_prob', 'draft_oracle_gate', 'target_oracle_gate_fixed', 'joint_confidence_product_fixed', 'confidence_agreement', 'rolling_accept_rate', 'rolling_accept_length']

def tree_predict(signals):
    if signals.top1_prob <= 0.4666:
        if signals.top1_prob <= 0.2703:
            if signals.target_top1_prob <= 0.8120:
                if signals.target_top1_prob <= 0.6549:
                    return "4,3"  # n=2845
                else:  # target_top1_prob > 0.6549
                    if signals.target_top1_prob <= 0.7401:
                        return "3,3"  # n=2352
                    else:  # target_top1_prob > 0.7401
                        return "4,3"  # n=2103
            else:  # target_top1_prob > 0.8120
                if signals.target_top1_prob <= 0.8909:
                    return "4,4"  # n=1822
                else:  # target_top1_prob > 0.8909
                    return "3,4"  # n=1152
        else:  # top1_prob > 0.2703
            if signals.target_top1_prob <= 0.8120:
                if signals.target_top1_prob <= 0.6548:
                    return "3,3"  # n=2478
                else:  # target_top1_prob > 0.6548
                    if signals.target_top1_prob <= 0.7401:
                        return "1,4"  # n=2263
                    else:  # target_top1_prob > 0.7401
                        return "3,3"  # n=2198
            else:  # target_top1_prob > 0.8120
                return "1,4"  # n=3335
    else:  # top1_prob > 0.4666
        if signals.target_top1_prob <= 0.8120:
            if signals.target_top1_prob <= 0.6554:
                if signals.target_top1_prob <= 0.6548:
                    return "3,4"  # n=4946
                else:  # target_top1_prob > 0.6548
                    return "3,4"  # n=30
            else:  # target_top1_prob > 0.6554
                if signals.joint_confidence_product_fixed <= 0.4931:
                    if signals.target_top1_prob <= 0.7401:
                        return "3,4"  # n=2258
                    else:  # target_top1_prob > 0.7401
                        return "2,5"  # n=1552
                else:  # joint_confidence_product_fixed > 0.4931
                    if signals.joint_confidence_product_fixed <= 0.5166:
                        return "2,5"  # n=504
                    else:  # joint_confidence_product_fixed > 0.5166
                        return "2,5"  # n=7291
        else:  # target_top1_prob > 0.8120
            if signals.joint_confidence_product_fixed <= 0.8858:
                if signals.joint_confidence_product_fixed <= 0.7793:
                    if signals.target_top1_prob <= 0.8909:
                        return "2,4"  # n=3906
                    else:  # target_top1_prob > 0.8909
                        return "2,5"  # n=2405
                else:  # joint_confidence_product_fixed > 0.7793
                    if signals.confidence_agreement <= 0.9155:
                        return "3,7"  # n=2492
                    else:  # confidence_agreement > 0.9155
                        return "3,7"  # n=1307
            else:  # joint_confidence_product_fixed > 0.8858
                if signals.top1_prob <= 0.9415:
                    return "3,7"  # n=260
                else:  # top1_prob > 0.9415
                    if signals.target_top1_prob <= 0.8911:
                        return "3,7"  # n=30
                    else:  # target_top1_prob > 0.8911
                        return "1,6"  # n=3838
