class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

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

    def buildTree(self, preorder, inorder):
        if not preorder:
            return None
        print(preorder,inorder)
        head = preorder[0]
        head_index = inorder.index(head)
        print(head_index)
        pre_length, next_length = head_index, len(preorder)-head_index-1
        result = TreeNode(head)
        result.left = self.buildTree(preorder[1:1+pre_length],inorder[:pre_length])
        result.right = self.buildTree(preorder[1+pre_length:],inorder[pre_length+1:])
        return result

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        a, b = 0, 1
        while n:
            a, b = b, a+b
            n -= 1
        return a%1000000007

    def Mininorder(self,numbers,left,right):
        for i in range(left+1,right):
            if numbers[i]<numbers[i-1]:
                return numbers[i]
        return numbers[right]
    def minArray(self, numbers):
        """
        :type numbers: List[int]
        :rtype: int
        """
        if len(numbers) == 0:
            return None
        if len(numbers) == 1 or numbers[0] < numbers[-1]:
            return numbers[0]
        left, right = 0, len(numbers)-1
        mid = left
        while numbers[left] >= numbers[right]:
            if right - left == 1:
                return numbers[right]
            mid = (left + right) // 2
            if numbers[mid] == numbers[left] and numbers[mid] == numbers[right]:
                return self.Mininorder(numbers,left,right)
            if numbers[mid] >= numbers[left]:
                left = mid
            else:
                right = mid
        return numbers[mid]

    def minArray(self,numbers):
        if len(numbers) == 0:
            return None
        if len(numbers) == 1 or numbers[0] < numbers[-1]:
            return numbers[0]
        left, right = 0, len(numbers)-1
        while left<right:
            mid = (left + right) // 2
            if numbers[mid] > numbers[right]:
                left = mid+ 1
            elif numbers[mid]< numbers[right]:
                right = mid
            else:
                right -= 1
        return numbers[left]

    # def hasPath(self,board,length,word,visited,row,col):
    #     hasPath = False
    #     if row>=0 and col>=0 and row<len(board) and col< len(board[0]) and board[row][col] == word[length] and not visited[row][col]:
    #         if length == len(word)-1:
    #             return True
    #         length += 1
    #         visited[row][col] = True
    #         hasPath = self.hasPath(board,length,word,visited,row-1,col) or self.hasPath(board,length,word,visited,row+1,col) or \
    #                   self.hasPath(board,length,word,visited,row,col-1) or self.hasPath(board,length,word,visited,row,col+1)
    #         if not hasPath:
    #             visited[row][col] = False
    #             length -=1
    #     return hasPath


    def exist(self, board, word):
        # 剑指offer12：矩阵中的路径
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        def hasPath(board,length,word,row,col):
            has_Path = False
            if row >= 0 and col >= 0 and row < len(board) and col < len(board[0]) and board[row][col] == word[length]:
                if length == len(word) - 1:
                    return True
                length += 1
                tmp, board[row][col] = board[row][col], ''
                has_Path = hasPath(board, length, word, row - 1, col) or hasPath(board, length, word, row+1, col) or hasPath(board, length, word, row, col - 1) or hasPath(board, length, word, row, col + 1)
                if not has_Path:
                    length -= 1
                board[row][col] = tmp
            return has_Path

        if board is None or len(board)<1 or not word:
            return False
        length = 0
        for row in range(len(board)):
            for col in range(len(board[0])):
                if hasPath(board,length,word,row,col):
                    return True
        return False

    def getDigit(self,num):
        sum = 0
        while num:
            sum += num%10
            num //=10
        return sum

    def check(self,m,n,k,row,col,visited):
        if row>=0 and col>=0 and row<m and col<n and self.getDigit(row)+self.getDigit(col)<=k and not visited[row][col]:
            return True
        return False

    def movingCountCore(self,m,n,k,row,col,visited):
        count = 0
        if self.check(m,n,k,row,col,visited):
            visited[row][col] = True
            count =1 + self.movingCountCore(m,n,k,row-1,col,visited) + self.movingCountCore(m,n,k,row+1,col,visited) + \
                     self.movingCountCore(m,n,k,row,col-1,visited) + self.movingCountCore(m,n,k,row,col+1,visited)
        return count

    def movingCount(self, m, n, k):
        """
        剑指offer13：机器人运动范围
        :type m: int
        :type n: int
        :type k: int
        :rtype: int
        """
        def getDigits(num):
            sum = 0
            while num:
                sum += num % 10
                num //= 10
            return sum
        # from queue import Queue
        q = []
        q.append((0,0))
        s = set()
        while q:
            x, y = q[0]
            q.pop(0)
            if (x,y) not in s and getDigits(x) +getDigits(y) <=k and 0<=x<m and 0<=y<n:
                s.add((x,y))
                q.append((x+1,y))
                q.append((x,y+1))
        return len(s)

    def cuttingRope(self, n):
        """
        剑指offer14：剪绳子
        :type n: int
        :rtype: int
        """
        if n<4:
            res =  n-1
        elif n==4:
            res =  n
        elif n%3 == 0:
            res =  3**(n//3)
        elif n%3 == 1:
            res =  4*3**(n//3-1)
        else:
            res =  2*3**(n//3)
        return res%(1e9+7)

    def hammingWeight(self, n):
        """
        剑指 Offer 15. 二进制中1的个数
        :type n: int
        :rtype: int
        """
        sum = 0
        while n:
            n = n&(n-1)
            sum += 1
        return sum

    def myPow(self, x, n):
        """
        剑指 Offer 16. 数值的整数次方
        :type x: float
        :type n: int
        :rtype: float
        """

        if n==0:
            return 1
        if n==1:
            return x
        if x==0:
            return 0
        is_nag = False
        if n<0:
            is_nag = True
            n = -n
        res = 1
        while n:
            if n&1:
                res *= x
            x = x*x
            n = n >> 1
        if is_nag:
            res = 1.0/res
        return res

    def printNumbers(self, n):
        """
        剑指 Offer 17. 打印从1到最大的n位数
        :type n: int
        :rtype: List[int]
        """
        res = 0
        while n:
            res = 10*res +9
            n -=1
        return range(1,res+1)

    def deleteNode(self, head:ListNode, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head.val == val:
            return head.next
        pre, next = head,head.next
        while next.val is not val:
            pre, next = next, next.next
        pre.next = next.next
        return head

    def exchange(self, nums):
        """
        剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums)<=1:
            return nums
        left, right = 0, len(nums)-1
        while left<right:
            while left<len(nums) and nums[left]&1:left +=1
            while right >=0 and not nums[right] % 2 &1: right -= 1
            if left<right:
                nums[left], nums[right] = nums[right], nums[left]
        return nums

    def isSubStructure(self, A, B):
        """
        剑指 Offer 26. 树的子结构
        :type A: TreeNode
        :type B: TreeNode
        :rtype: bool
        """
        def isSub(A,B):
            if not B:
                return True
            if not A:
                return False
            if A.val is not B.val:
                return False
            return isSub(A.left,B.left) and isSub(A.right,B.right)

        if not B or not A:
            return False

        return isSub(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)

    def mirrorTree(self, root):
        """
        剑指 Offer 27. 二叉树的镜像
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        root.left, root.right = self.mirrorTree(root.right), self.mirrorTree(root.left)
        return root

    def isSymmetric(self, root):
        """
        剑指 Offer 28. 对称的二叉树
        :type root: TreeNode
        :rtype: bool
        """
        def Symmertric(A,B):
            if not A and not B:
                return True
            if not A or not B or A.val != B.val:
                return False
            return Symmertric(A.left,B.right) and Symmertric(A.right,B.left)
        if not root:
            return True
        return Symmertric(root.left,root.right)

    def spiralOrder(self, matrix):
        """
        剑指 Offer 29. 顺时针打印矩阵
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res = []
        def printCircle(start):
            end_row = len(matrix)-1-start
            end_col = len(matrix[0])-1-start
            for col in range(start,end_col+1):
                res.append(matrix[start][col])
            if end_row>start:
                for row in range(start+1,end_row+1):
                    res.append(matrix[row][end_col])
                if end_col>start:
                    for col in range(end_col-1,start-1,-1):
                        res.append(matrix[end_row][col])
                    if end_row-1>start:
                        for row in range(end_row-1,start,-1):
                            res.append(matrix[row][start])
            return
        if not matrix:return 0
        start = 0
        while start*2<len(matrix) and start*2<len(matrix[0]):
            printCircle(start)
            start+=1
        return res

    class MinStack(object):
        '''
        剑指 Offer 30. 包含min函数的栈
        '''

        def __init__(self):
            """
            initialize your data structure here.
            """
            self.stack, self.min_stack = [], []

        def push(self, x):
            """
            :type x: int
            :rtype: None
            """
            self.stack.append(x)
            if len(self.min_stack)==0:
                self.min_stack.append(x)
            else:
                self.min_stack.append(min(x,self.min_stack[-1]))

        def pop(self):
            """
            :rtype: None
            """
            self.min_stack.pop()
            return self.stack.pop()

        def top(self):
            """
            :rtype: int
            """
            return self.stack[-1]

        def min(self):
            """
            :rtype: int
            """
            return self.min_stack[-1]

    def validateStackSequences(self, pushed, popped):
        """
        剑指 Offer 31. 栈的压入、弹出序列
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        if (not pushed and not popped)  or (len(pushed) == 1 and pushed[0] == popped[0]):
            return True
        if not pushed or not popped:
            return False
        stack = [pushed[0]]
        push_index, pop_index = 1, 0
        while push_index<len(pushed) and pop_index < len(popped):
            while push_index<len(pushed) and (not stack or stack[-1]!=popped[pop_index] ):
                stack.append(pushed[push_index])
                push_index+=1
            while len(stack)>0 and stack[-1] == popped[pop_index] :
                stack.pop()
                pop_index+=1
        if pop_index == len(popped):
            return True
        return False

    def levelOrder(self, root):
        """
        剑指 Offer 32 - I. 从上到下打印二叉树
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res, queue = [], [root]
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return res

    def levelOrder(self, root):
        """
        剑指 Offer 32 - I. 从上到下打印二叉树2
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res, res_level, queue = [], [], [root]
        toPrint, next_leval = 1, 0
        while queue:
            node = queue.pop(0)
            toPrint -=1
            res_level.append(node.val)
            if node.left:
                queue.append(node.left)
                next_leval +=1
            if node.right:
                queue.append(node.right)
                next_leval+=1
            if not toPrint:
                res.append(res_level)
                res_level = []
                toPrint = next_leval
                next_leval = 0
        return res

    def levelOrder(self, root):
        """
        剑指 Offer 32 - III. 从上到下打印二叉树 III
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res, res_level, queue = [], [], [root]
        toPrint, next_leval = 1, 0
        need_reverse = False
        while queue:
            node = queue.pop(0)
            toPrint -=1
            res_level.append(node.val)
            if node.left:
                queue.append(node.left)
                next_leval +=1
            if node.right:
                queue.append(node.right)
                next_leval+=1
            if not toPrint:
                if need_reverse:
                    res_level = res_level[::-1]
                need_reverse = not need_reverse
                res.append(res_level)
                res_level = []
                toPrint = next_leval
                next_leval = 0
        return res

    def verifyPostorder(self, postorder):
        """
        剑指 Offer 33. 二叉搜索树的后序遍历序列
        :type postorder: List[int]
        :rtype: bool
        """
        if not postorder or len(postorder)==1:
            return True
        root = postorder[-1]
        is_left = True
        left_end = 0
        for index, num in enumerate(postorder[:-1]):
            if is_left:
                if num<root:
                    pass
                else:
                    left_end = index
                    is_left=False
            else:
                if num<root:
                    return False
        return self.verifyPostorder(postorder[:left_end]) and self.verifyPostorder(postorder[left_end:-1])

    def pathSum(self, root, sum):
        """
        剑指 Offer 34. 二叉树中和为某一值的路径
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        if root.val == sum and not root.left and not root.right:
            return [[root.val]]
        next_leavel = []
        left_path, right_path = self.pathSum(root.left, sum - root.val), self.pathSum(root.right, sum - root.val)
        if left_path:
            next_leavel.extend(left_path)
        if right_path:
            next_leavel.extend(right_path)
        if next_leavel:
            res = []
            for path in next_leavel:
                tmp = [root.val]
                tmp.extend(path)
                res.append(tmp)
            return res
        return []



solution = Solution()
print(solution.validateStackSequences([1],[1]))
test = []


# import heapq
# heap = [5,1,6,8,2,5,4,7,243,325,16,4,6312,7]
# heapq.heapify(heap)
# print(heap)
# heapq.heappush(heap,1)
# print(heapq.heappop(heap))