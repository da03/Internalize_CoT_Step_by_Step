import re
def is_correct(ans, pred_ans):
    ans = ans.replace('#', '').strip().replace(' ', '')
    pred_ans = pred_ans.replace('#', '').strip().replace(' ', '')
    try:
        # Remove leading zeros in numbers before evaluating
        def remove_leading_zeros(expr):
            # Replace any number with leading zeros (but not just '0')
            return re.sub(r'\b0+([1-9][0-9]*)\b', r'\1', expr)
        
        ans = remove_leading_zeros(ans)
        pred_ans = remove_leading_zeros(pred_ans)
        #import pdb; pdb.set_trace()
        # evaluate ans and pred_ans
        ans_result = eval(ans)
        pred_ans_result = eval(pred_ans)
        print (ans_result, pred_ans_result)
        return ans_result == pred_ans_result
    except:
        return ans == pred_ans


is_correct('0 9 8 + 0 0 1 * 0 5 5 + 0 2 2 - 0 8 4 + 0 2 0 - 0 9 9 + 0 5 2', '0 5 2 - 0 2 0 + 0 9 9 - 0 2 2 - 0 0 1 + 0 9 8 - 0 5 5 - 0 8 4')
