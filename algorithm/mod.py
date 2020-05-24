import random
def mod_equal(nums,k):
    pre_sum = [nums[0]%k]
    for i in range(1,len(nums)):
        pre_sum.append((pre_sum[-1]+nums[i])%k)
    is_second = [False for _ in range(k)]
    for t in range(len(pre_sum)):
        if is_second[pre_sum[t]]:
            for j in range(t):
                if pre_sum[t] == pre_sum[j]:
                    return j+1,t
        else:
            is_second[pre_sum[t]] = True


print(mod_equal([1,2,3,4,5,6,7,],5))



