import random
import numpy as np
from utils.calcs import semantic_similarity


def get_line_level_knowledge_mask(
    wrong_code: str,
    correct_code: str,
    prefix_prompt: str,
    delete_context_mask_len: int,
    random_mask: bool,
    random_mask_ratio: float,
    all_context_mask_len: int,
    post_context_only: bool,
):
    line_level_knowledge_mask4wrong_code = get_line_level_knowledge_mask4wrong_code(
        wrong_code,
        correct_code,
        delete_context_mask_len,
        random_mask,
        random_mask_ratio,
        all_context_mask_len,
        post_context_only,
    )

    line_level_knowledge_mask4prefix_prompt = [0] * len(prefix_prompt.splitlines())

    line_level_knowledge_mask = (
        line_level_knowledge_mask4prefix_prompt
        + line_level_knowledge_mask4wrong_code
    )

    semantic_relatedness = semantic_similarity(correct_code)
    line_level_knowledge_mask = spread_knowledge_mask_by_semantic_relatedness(
        line_level_knowledge_mask, semantic_relatedness
    )

    return line_level_knowledge_mask


def get_line_level_knowledge_mask4wrong_code(
    wrong_code: str,
    correct_code: str,
    delete_context_mask_len: int,
    random_mask: bool,
    random_mask_ratio: float,
    all_context_mask_len: int,
    post_context_only: bool,
):
    """
    Calculate the edit operations between two multi-line strings wrong_code and correct_code, and return a line level knowledge mask for correct_code,
    marking modified lines and only the post context of operations.
    @param wrong_code: The wrong code.
    @param correct_code: The correct code.
    @param delete_context_mask_len: The length of context to be masked for deletion.
    @param random_mask: Whether to apply random mask.
    @param random_mask_ratio: The ratio of random mask.
    @param all_context_mask_len: The length of context to be masked for d, i, r.
    @param post_context_only: Whether to only mask post context.
        if True, only mask post context.
        if False, mask both pre and post context.
    """
    wrong_lines = wrong_code.splitlines()
    correct_lines = correct_code.splitlines()

    m, n = len(wrong_lines), len(correct_lines)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if wrong_lines[i - 1] == correct_lines[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Knowledge mask initialization
    line_level_knowledge_mask = [0] * n
    i, j = m, n

    while i > 0 and j > 0:
        if wrong_lines[i - 1] == correct_lines[j - 1]:
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            mark_in_line_level_knowledge_mask(
                j,
                n,
                line_level_knowledge_mask,
                delete_context_mask_len + all_context_mask_len,
                post_context_only,
            )  # Mark post context of deleted line
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            line_level_knowledge_mask[j - 1] = 1  # Mark inserted line
            mark_in_line_level_knowledge_mask(
                j, n, line_level_knowledge_mask, all_context_mask_len, post_context_only
            )  # Mark post context of inserted line
            j -= 1
        else:
            line_level_knowledge_mask[j - 1] = 1  # Mark replaced line
            mark_in_line_level_knowledge_mask(
                j, n, line_level_knowledge_mask, all_context_mask_len, post_context_only
            )  # Mark post context of replaced line
            i, j = i - 1, j - 1

    while i > 0:
        # For remaining deletion, only mark post context in b
        mark_in_line_level_knowledge_mask(
            j,
            n,
            line_level_knowledge_mask,
            delete_context_mask_len + all_context_mask_len,
            post_context_only,
        )
        i -= 1

    while j > 0:
        line_level_knowledge_mask[j - 1] = 1  # Mark remaining inserted lines
        mark_in_line_level_knowledge_mask(
            j,
            n,
            line_level_knowledge_mask,
            all_context_mask_len + all_context_mask_len,
            post_context_only,
        )  # Mark post context of remaining inserted lines
        j -= 1

    # Apply random mask if enabled
    if random_mask:
        for k in range(n):
            if (
                line_level_knowledge_mask[k] == 0
                and random.random() < random_mask_ratio
            ):
                line_level_knowledge_mask[k] = 1

    return line_level_knowledge_mask


def mark_in_line_level_knowledge_mask(
    current_j, total_j, mask, context_len, post_context_only=False
):
    """
    Mark the context of a line in knowledge mask of b.
    """
    if post_context_only:
        start = current_j
    else:
        start = max(0, current_j - context_len)
    end = min(total_j, current_j + context_len)
    for k in range(start, end):
        mask[k] = 1


def spread_knowledge_mask_by_semantic_relatedness(
    line_level_knowledge_mask, semantic_relatedness
):
    """
    Spread knowledge mask by semantic relatedness.
    """
    n = len(line_level_knowledge_mask)
    new_knowledge_mask = [0] * n

    for i in range(n):
        if line_level_knowledge_mask[i] == 1:
            new_knowledge_mask[i] = 1
            for j in range(n):
                if j == i:
                    continue
                if semantic_relatedness[i][j] == 1:
                    new_knowledge_mask[j] = max(
                        new_knowledge_mask[j] + (1 / (np.log(np.abs(j - i) + 1))),
                        1
                    )
                else:
                    break

    return new_knowledge_mask


def get_input_id_level_knowledge_mask(token_ids, knowledge_mask):
    """
    Create mask for text according to knowledge mask.
    """

    mask = [0] * len(token_ids)
    '''
    for `tokenizer.add_eos_token = True`
    so there is </s> at the end of token_ids
    token_ids[-1] = tokenize(</s>)
    token_ids[-2] = tokenize('/n') = 13
    need an extra 0 at the end of knowledge_mask
    and we want to train the model to predict </s> token, so knowledge_mask[-1] = 1
    '''
    knowledge_mask += [1]

    line_index = 0
    
    import random

    for i, token_id in enumerate(token_ids):
        if knowledge_mask[line_index] != 0:
            mask[i] = knowledge_mask[line_index]
        else:
            mask[i] = float(random.randint(0, 80)) / 100

        if token_id == 13:
            line_index += 1

    return mask


