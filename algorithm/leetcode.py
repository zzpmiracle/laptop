def longestPalindrome(s):
    """
    :type s: str
    :rtype: str
    """
    max_len, result = float("-inf"), ""
    for i in range(len(s)):
        for j in range(i + 1, len(s)+1):
            if s[i:j] == s[i:j][::-1]:
                if j - i > max_len:
                    max_len = j - i
                    result = s[i:j]
    return result
def DP(s):
    if len(s) == 0:
        return ''
    dp = [[False for _ in range(len(s))] for __ in range(len(s))]
    left, right = 0, 0
    for i in range(len(s)-1,-1,-1):
        dp[i][i] = True
        for j in range(i+1,len(s)):
            dp[i][j] = s[i]==s[j] and (j-i<3 or dp[i+1][j-1])
            if dp[i][j] and right-left < j-i:
                left = i
                right = j
    return s[left:right+1]


import time

class Solution:
    class Solution:
        def threeSum(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
            nums.sort()
            N, result = len(nums), []
            for i in range(N):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                target = nums[i] * -1
                s, e = i + 1, N - 1
                while s < e:
                    if nums[s] + nums[e] == target:
                        result.append([nums[i], nums[s], nums[e]])
                        s = s + 1
                        while s < e and nums[s] == nums[s - 1]:
                            s = s + 1
                    elif nums[s] + nums[e] < target:
                        s = s + 1
                    else:
                        e = e - 1
            return result
    def fourSum(self, nums,target: int):
        if len(nums)==0:
            return []
        nums.sort()
        N, res = len(nums), []
        for num1 in range(N):
            if num1>0 and nums[num1] == nums[num1-1]:
                continue
            else:
                target_1 = target - nums[num1]
                for num2 in range(num1+1,N):
                    if nums>num1+1 and nums[num2] == nums[num2-1]:
                        continue
                    else:
                        target_2 = target_1 - nums[num2]
                        num3, num4 = num2+1, N-1
                        while(num3<num4):
                            if nums[num3]+nums[num4]== target_2:
                                res.append([nums[num1],nums[num2],nums[num3],nums[num4],])
                                while(num3<num4 and nums[num3]==nums[num3+1]):
                                    num3+=1
                                while(num3 < num4 and nums[num4] == nums[num4 - 1]):
                                    num4-=1
                                num3+=1
                                num4-=1
                            elif nums[num3]+nums[num4]<target_2:
                                num3+=1
                            else:
                                num4-=1
        return res

    def divide(self, dividend: int, divisor: int):
        sign = -1 if (dividend^divisor)<0 else 1
        a,b = abs(dividend), abs(divisor)
        if a<b:
            return 0
        i = 0
        while a>= b:
            b = b<<1
            i = i+1
        res = (1<<(i-1)) + self.divide(a-(b>>1),abs(divisor))
        res *= sign
        return min(res,(1<<31)-1)

    def is_Substring(self,s,word_list,word_dic,word_len):
        s_list = []
        print(s)
        for i in range(len(s)//word_len):
            word = s[i*word_len:(i+1)*word_len]
            if word in word_dic:
                s_list.append(word_dic[word])
            else:
                return False
        if sorted(s_list) == word_list:
            return True
        else:return False
    def findSubstring(self, s: str, words):
        res = []
        word_dic, word_list = {}, []
        word_len = len(words[0])
        all_len = word_len*len(words)
        i = 0
        for word in words:
            if word not in word_dic:
                word_dic[word] = i
                i += 1
            word_list.append(word_dic[word])
        word_list.sort()
        for i in range(len(s)- all_len+1):
            s_tmp = s[i:i+all_len]
            if self.is_Substring(s_tmp,word_list,word_dic,word_len):
                res.append(i)
        return res

    def nextPermutation(self, nums) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        now_max, index= nums[-1], len(nums)-1
        is_MAX = False
        for num in range(len(nums)-2,-1,-1):
            if num == 0 and nums[0]>=nums[1]:
                is_MAX = True

            if nums[num] < now_max:
                index = num
                break
            else:
                now_max = nums[num]
        if is_MAX:
            nums.reverse()
        else:
            bigger_num = len(nums)-1
            while nums[bigger_num]<= nums[index]:
                bigger_num-=1
            nums[index], nums[bigger_num] =nums[bigger_num], nums[index]
            index, end = index+1, len(nums)-1
            while index < end:
                nums[index], nums[end] = nums[end], nums[index]
                index, end = index+1, end-1
        return nums

    def longestValidParentheses(self, s: str) -> int:

        if len(s)==0:
            return 0
        now_max, now, remain= 0, 0, 0
        for c in s:
            if c == '(':
                remain+=1
            elif c == ')':
                if remain>0:
                    remain-=1
                    now+=2
                    if now > now_max:
                        now_max = now
                else:
                    now, remain= 0, 0
        return now_max

solution = Solution()
print(solution.longestValidParentheses("()(()"))
import heapq
heap = [5,1,6,8,2,5,4,7,243,325,16,4,6312,7]
heapq.heapify(heap)
print(heap)
heapq.heappush(heap,1)
print(heapq.heappop(heap))