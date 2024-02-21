"""Linear Least Squares Curve Fitting (Regression) Simulations - II
"""
from Regression import LeastSquaresCurveFitting
import xlrd
import numpy as np
import matplotlib.pyplot as plt


class Cross_Validation:
     def __init__(self, data, n=18):
          self.data = data
          self.l = [] # self.l[i] has all the instances of LeastSquaresCurveFitting of degree i+2 
          for i in range (n):
               self.l.append([])
               for j in range (1, len(data)): # Y_Set i has an index of i-1
                    self.l[i].append(LeastSquaresCurveFitting(data = [self.data[0],self.data[j]], y_set = j , dim = i+2))
##          print(len(self.data[0]))                                                           
          self.RSS = self.RSS_matrix(n)
          self.R2 = self.R2_matrix(n)
          print(self.R2)
          self.stats_R2 = self.R2_stats(n)
          self.print_R2(n)
          s = input("Press Enter to continue")
          if s == "":  
               self.stats_plot(n)
          a = eval(input("Enter model order:"))
          self.stats_post_order_selection(a)
          b = eval(input("Enter data set no.:"))
          print("The equation of the line is:", self.l[a-1][b-1].equation)
          self.l[a-1][b-1].plot()
          self.final_plot(a,b)
          
          
          
     def RSS_matrix(self,n):
          """ l[i][j][k]
               i = training set i+1
               j = testing set j+1
               k = polynomial of degree k+1
                    when k == 0, we get a line y = mx + c
               l[0][0][0] has been trained with set 1, tested with set 1 and is of degree 2 (a line)
          """
          l = []
          for i in range (len(self.data)-1):
               l.append([])
               for j in range (len(self.data)-1):
                    l[i].append([])
                    for k in range (n):
                         a,b = self.l[k][i].compute_e_and_TLSE(self.data[j+1])
                         l[i][j].append(b)
          return l

     def R2_matrix(self,n):
          """ l[i][j][k]
               i = training set i+1
               j = testing set j+1
               k = polynomial of degree k+1
                    when k == 0, we get a line y = mx + c
               l[0][0][0] has been trained with set 1, tested with set 1 and is of degree 2 (a line)
          """
          l = []
          for i in range (len(self.data)-1):
               l.append([])
               for j in range (len(self.data)-1):
                    l[i].append([])
                    for k in range (n):
                         rss_mean = self.RSS[i][j][k]/len(self.data[0])
                         var_y = np.var(np.array(self.data[j+1]))
                         l[i][j].append(1-(rss_mean/var_y))
          return l

     def print_R2(self,n):
          for k in range (n):
               print("n =", k+1)
               for i in range (len(self.data)-1):
                    for j in range (len(self.data)-1):
                         print(format(self.R2[i][j][k], ".7f"), end = " ")
                    print()
               
               print("Mean:", format(self.stats_R2[k][0], ".5f"))
               print("Min:", format(self.stats_R2[k][1], ".5f"))
               print("Max:", format(self.stats_R2[k][2], ".5f"))
               print("Stdev:", format(self.stats_R2[k][3], ".5f"))
               print()
               print()
                    
     def R2_stats(self,n,include=False):
          """ output:
               index 0: mean
               index 1: min
               index 2: max
               index 3: stdev
          """
          l = []
          for k in range (n):
               l.append([])
               l1 = []
               for i in range (len(self.data)-1):
                    for j in range (len(self.data)-1):
                         if (i != j) or include:
                              l1.append(self.R2[i][j][k])
               l[k].append(sum(l1)/len(l1))
               l[k].append(min(l1))
               l[k].append(max(l1))
               l[k].append(np.std(np.array(l1)))
          return l


     def stats_plot(self,n):
          x = []
          for i in range (n):
               x.append(i+1)
          x = np.array(x)
          l = []
          for i in range(len(self.stats_R2[0])):
               l.append([])
               if i != (len(self.stats_R2[0])-1):
                    for j in range(len(self.stats_R2)):
                         l[i].append(self.stats_R2[j][i]+i)
               else:
                    for j in range(len(self.stats_R2)):
                         l[i].append((2*self.stats_R2[j][i])+i)
          l = np.array(l)
          label = ["Mean", "Min", "Max", "Stdev"]
          for i in range(len(self.stats_R2[0])):
               plt.plot(x, l[i], marker = 'o', label = label[i])
          plt.legend()
          plt.grid()
          plt.xlabel("Model Order")
          plt.title("Stats on R2")
          print("Close the graph to continue")
          plt.show()
          
     def stats_post_order_selection(self,a):
          """ l is a 2D matrix corresponding to model oreder a
               l[i][j]:
               i = training set i+1
               j = testing set j+1

               l_stats: 2D matrix which contains some stats on the elements of l
               l_stats[i][j]:
               i = training set i+1
               j:
                    0: mean
                    1: min
                    2: max
                    3: stdev
          """
          l = []
          l_stats = []
          for i in range (len(self.data)-1):
               l.append([])
               l_stats.append([])
               for j in range (len(self.data)-1):
                    l[i].append(self.R2[i][j][a-1])
               l_stats[i].append(sum(l[i])/len(l[i]))
               l_stats[i].append(min(l[i]))
               l_stats[i].append(max(l[i]))
               l_stats[i].append(np.std(np.array(l[i])))
          for i in range (len(l_stats)):
##               for j in range (len(l_stats[i])):
               print("Stats on R2 of model trained on data set:", i+1)
               print("Mean:", format(l_stats[i][0], ".5f"))
               print("Min:", format(l_stats[i][1], ".5f"))
               print("Max:", format(l_stats[i][2], ".5f"))
               print("Stdev:", format(l_stats[i][3], ".5f"))
               print()
##          print(np.array(l_stats))
##          print(np.array(l))

          s = input("Enter to continue")
          if s == "":
          # Plotting the result
               x = []
               for i in range (len(self.data)-1):
                    x.append(i+1)
               x = np.array(x)
               l1 = []
               for i in range(len(l_stats[0])):
                    l1.append([])
                    if i != (len(l_stats[0])-1):
                         for j in range(len(l_stats)):
                              l1[i].append(l_stats[j][i]+i)
                    else:
                         for j in range(len(l_stats)):
                              l1[i].append((2*l_stats[j][i])+i)
               l1 = np.array(l1)
               label = ["Mean", "Min", "Max", "Stdev"]
               for i in range(len(l_stats[0])):
                    plt.plot(x, l1[i], marker = 'o', label = label[i])
               plt.legend(loc="right")
               plt.grid()
               plt.title("Model Order "+str(a))
               plt.xlabel("Data Set no.")
               print("Close the graph to continue")
               plt.show()

     def final_plot(self,a,b):
          for i in range(1,len(self.data)):
               if i != b:
                    plt.scatter(self.data[0], self.data[i])
          self.l[a-1][b-1].plot()
          print("The program has ended")
          
          



DATA_SETS = 6
LOC = r"C:\Users\Shubham\AppData\Local\Programs\Python\Python39\Workspace\Linear_Algebra\2\line_data.xls" # name of the excel file
##LOC = r"C:\Users\shubha\AppData\Local\Programs\Python\Python37\Workspace\Linear_Algebra\1\line_data.xls"


def read_excel(loc):
     # To open Workbook
     wb = xlrd.open_workbook(loc)
     sheet = wb.sheet_by_index(0)

     # Create data by reading the columns of the file
     x = [] 
     for i in range (DATA_SETS):
          x.append(sheet.col_values(i)[1:22])
          
     return x

##for u in x:
##     print(u)



a = Cross_Validation(data = read_excel(LOC))
##a.l[0][1].plot()
