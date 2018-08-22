# -*- coding: utf-8 -*-
#--enable-unicode=ucs4

import sys
import os
import numpy as np
import pandas as pd
import sklearn as skl
import scipy
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet


from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

#Definición de funciones que se utilizarán en la ejecución del algoritmo
def calculate_new_centroids(centroids,clusters):
    """
    "clusters" es una matriz no cuadrada donde se encuentran los elementos (en ejecución) de cada cluster,
    para calcular los nuevos centroides se tiene que sacar un promedio por cada elemento. Por ejemplo, para esta
    implementación se tomó en cuenta solo 2 dimensiones (x,y), entonces, esta función hace la suma de todos los "x"
    y por separado todos los "y", finalmente divide entre el numero de elementos en el cluster "K" y esos serían
    los elementos para el nuevo centroide "K".
    """
    x_sum,y_sum=0,0
    new_centroids=np.zeros(np.shape(centroids))
    element_shape=np.shape(clusters[0])

    for i in range(0,len(clusters)):
        for element_n in clusters[i]:
        	x=np.asarray(element_n)
        	y=np.asarray(new_centroids[i])
        	new_centroids[i]=x+y

    for i in range(0,len(new_centroids)):
    	for j in range(0,len(new_centroids[i])):
            new_centroids[i][j]=new_centroids[i][j]/len(clusters[i])

    return  new_centroids

def distances_matrix(points_matrix,centroids):
    """Función que retorna una matriz de K columnas (las columnas representan el numero de centroides requeridos)
    y N elementos/filas, se calcula la distancia euclidiana de cada elemento en points_matrix hacia cada uno de los centroides
    """
    distance_matriz=np.zeros([len(points_matrix),len(centroids)])
    i,j=0,0
    for po_m in points_matrix:
        for cent in centroids:
            var_index_centroid= np.sum((po_m-cent)**2,axis=None)
            distance_matriz[i,j]=var_index_centroid
            j+=1
        i+=1
        j=0
    return distance_matriz

def olds_vs_news(olds,news):
    """
    Compara la distancia de una matriz de centroides de una iteración anterior y de centroies "recien calculados".
    El valor de comparación puede ser modificable dependiendo el dataset, en este caso se utilizó el valor de 0.001
    para todos los dataset.
    """
    flag=False
    v_flag=np.zeros([len(news)])
    ceros=np.zeros([len(news)])
    if(len(olds)==len(news)):
        for i in range(0,len(olds)):
            distance=np.sum((olds[i]-news[i])**2,axis=0)
            if(distance<0.000001):
                v_flag[i]=1
        if(np.sum(v_flag+ceros,axis=0)==len(v_flag)):
            return True
        else:
            return False

def matriz_indices(f,c):
    """
    #Crea matriz de indices dimenciones FxC. Ejemplo, si f=2 y c=3 entonces, matriz_indices=[[0,1,2],[0,1,2]]
    """
    matriz_indices=[]
    for i in range(0,f):
        columna=[]
        for j in range(0,c):
            columna.append(j)
        matriz_indices.append(columna)
    return (np.asarray(matriz_indices))

def sort_index(distances_m,matriz_de_indices):
    #Función que acomoda los indices de la matriz de indices y coloca la distancia mínima de cada fila en la primer columna
    """
    ejemplo: distances_m=[[0.333,0.111],[8,9]] y matriz_de_indices=[[0,1],[0,1]]
    entonces las matrices después del proceso quedarán:
    distances_m=[[0.111,0.333],[8,9]] y matriz_de_indices=[[1,0],[0,1]]

    para cada fila colocará en la primer columna el valor menor y dependiendo a que indice pertenecia originalmente la 
    matríz de indices se modificará de igual manera
    """
    m=0
    for sq_d in distances_m:
        for i in range(0,len(sq_d)):
            for j in range(0,len(sq_d)):
                if(sq_d[i]<sq_d[j] and i!=j):
                    sq_d_respaldo=sq_d[j]
                    sq_d[j]=sq_d[i]
                    sq_d[i]=sq_d_respaldo
                    sq_d_respaldo=matriz_de_indices[m,j]
                    matriz_de_indices[m,j]=matriz_de_indices[m,i]
                    matriz_de_indices[m,i]=sq_d_respaldo
        m+=1

def iniciar_clusters(data,index_matrix,k):
    clusters=[]
    for i in range(0,k):
        clusters.append(list([]))

    for i in range(0,len(clusters)):
        for j in range(0,len(data)):
            if(index_matrix[j,0]==i):
                clusters[i].append(list(data[j]))
            
    return clusters

def get_n_classes(file_path):
    """
    Obtiene la última columna de cada archivo, donde están las "etiquetas de clase", se hace una lista para después
    borrar los elementos repetidos. La función retorna la cantidad de elementos no repetidos de la lista de etiquetas, así
    se sabe el número de grupos de cada dataset.
    """
    file=open(file_path)
    clases=[]
    flag=0
    index=0
    for row in file:
        row=row.split()
        if(flag==0):
            index=len(row)
            i=1
        clases.append(row[index-1])
    clases=list(set(clases))
    clases=np.asarray(clases)
    cl=clases.astype(np.float)
    return len(cl)

def get_data(path):
    """
    Lee las primeras 2 columnas del archivo en ejecución
    """
    data=[]
    file=open(path)
    flag=0
    length_f=0
    for row in file:
        row=row.split()
        if(flag==0):
            length_f=len(row)
            flag=1
        data.append(row[0:2])
    data=np.asarray(data)
    return data.astype(np.float)

def get_tags(path):
    """En ésta función obtiene la última columna, donde en todos los archivos se sabe que están las etiquetas representativas de los grupos
    para traerlas todas y hacer una lista de ellas
    """
    data=[]
    file=open(path)
    flag=0
    length_f=0
    for row in file:
        row=row.split()
        if(flag==0):
            length_f=len(row)
            flag=1
        data.append(row[length_f-1])
    data=np.asarray(data)
    return data.astype(int)

#Fin funciones k-means

def average_linkage(p,colors):
    path=p
    folders=os.listdir(path)
    folders=np.asarray(folders)
    files=[]#inicializa lista para los nombres de los archivos pertenecientes a cada carpeta
    for f in folders:
        provisional_path=path+"/"+f#Se inicializa variable con la dirección anterior de la variable 'path' + el nombre de una carpeta 'x' dentro de StopSearch_2011_2017
        n_classes=get_n_classes(provisional_path)
        files.append([provisional_path,n_classes])

    files=np.asarray(files)
    count=0
    file_index=0
    
    results=[]
    for i in range(0,len(files)):
        X=np.asarray(get_data(files[i][0]))
        file_tags=get_tags(files[i][0])
        K=int(files[i][1])
        X=StandardScaler().fit_transform(X)
        Hclustering=AgglomerativeClustering(n_clusters=K,affinity='euclidean',linkage='average')
        y_hc=Hclustering.fit_predict(X)
        results.append(adjusted_rand_score(file_tags,y_hc)) 
        plt.subplots()
        for j in range(0,K):
            plt.scatter(X[y_hc==j,0],X[y_hc==j,1],c=colors[j])
        plt.title("average_"+folders[i])
        plt.savefig("average_"+folders[i]+".png")
    return results
      
def single_linkage(p,colors):
    path=p
    folders=os.listdir(path)
    folders=np.asarray(folders)
    files=[]#inicializa lista para los nombres de los archivos pertenecientes a cada carpeta
    #print(folders)
    for f in folders:
        provisional_path=path+"/"+f#Se inicializa variable con la dirección anterior de la variable 'path' + el nombre de una carpeta 'x' dentro de StopSearch_2011_2017
        n_classes=get_n_classes(provisional_path)
        files.append([provisional_path,n_classes])

    files=np.asarray(files)
    count=0
    file_index=0
    
    results=[]
    for i in range(0,len(files)):
        X=np.asarray(get_data(files[i][0]))
        file_tags=get_tags(files[i][0])
        K=int(files[i][1])
        X=StandardScaler().fit_transform(X)
        Hclustering=AgglomerativeClustering(n_clusters=K,affinity='euclidean',linkage='ward')
        y_hc=Hclustering.fit_predict(X)
        results.append(adjusted_rand_score(file_tags,Hclustering.labels_))
        plt.subplots()
        for j in range(0,K):
            plt.scatter(X[y_hc==j,0],X[y_hc==j,1],c=colors[j])
        plt.title("single_"+folders[i])
        plt.savefig("single_"+folders[i]+".png")
    
    return results

def complete_linkage(p,colors):
    path=p
    folders=os.listdir(path)
    folders=np.asarray(folders)
    files=[]#inicializa lista para los nombres de los archivos pertenecientes a cada carpeta
    for f in folders:
        provisional_path=path+"/"+f#Se inicializa variable con la dirección anterior de la variable 'path' + el nombre de una carpeta 'x' dentro de StopSearch_2011_2017
        n_classes=get_n_classes(provisional_path)
        files.append([provisional_path,n_classes])

    files=np.asarray(files)
    count=0
    file_index=0
    
    results=[]
    for i in range(0,len(files)):
        X=np.asarray(get_data(files[i][0]))
        file_tags=get_tags(files[i][0])
        K=int(files[i][1])
        X=StandardScaler().fit_transform(X)
        Hclustering=AgglomerativeClustering(n_clusters=K,affinity='euclidean',linkage='complete')
        y_hc=Hclustering.fit_predict(X)
        results.append(adjusted_rand_score(file_tags,Hclustering.labels_))
        plt.subplots()
        for j in range(0,K):
            plt.scatter(X[y_hc==j,0],X[y_hc==j,1],c=colors[j])
        plt.title("complete_"+folders[i])
        plt.savefig("complete_"+folders[i]+".png")
    
    return results

def kmeans(p,colors):
    path=p
    folders=os.listdir(path)#Se crea una lista con los nombres de todas las carpetas dentro de StopSearch_2011_2017
    folders=np.asarray(folders)
    files=[]#inicializa lista para los nombres de los archivos pertenecientes a cada carpeta
    #print(folders)
    for f in folders:
        provisional_path=path+"/"+f#Se inicializa variable con la dirección anterior de la variable 'path' + el nombre de una carpeta 'x' dentro de StopSearch_2011_2017
        n_classes=get_n_classes(provisional_path)
        files.append([provisional_path,n_classes])

    files=np.asarray(files)
    count=0
    file_index=0
    
    results=[]
    final_tags=[]
    for i in range(0,len(files)):
        file_name=folders[i]
        X=np.asarray(get_data(files[i][0]))
        file_tags=get_tags(files[i][0])
        X=StandardScaler().fit_transform(X)
        #print(X)
        K=int(files[i][1])
        X_shape=np.shape(X)
        k_index=np.random.choice(int(X_shape[0]),int(K),replace=False)#Selecciona indices aleatorios de elementos de la matriz original "X"
        centroids=X[k_index,:]#Asigna centroides aleatorios
        matriz_indx=matriz_indices(len(X),len(centroids))
        matriz_distancias=distances_matrix(X,centroids)
        sort_index(matriz_distancias,matriz_indx)
        #print(matriz_distancias)
        clusters=iniciar_clusters(X,matriz_indx,K)
        #print(clusters)
        old_centroids=centroids
        new_centroids=calculate_new_centroids(centroids,clusters)
        while(olds_vs_news(old_centroids,new_centroids)):
            matriz_indx=matriz_indices(len(X),len(centroids))
            matriz_distancias=distances_matrix(X,centroids)
            sort_index(X,matriz_indx)
            clusters=iniciar_clusters(X,matriz_indx,K)
            old_centroids=new_centroids
            new_centroids=calculate_new_centroids(centroids,clusters)

        for element in matriz_indx:
            final_tags.append(element[0])

        results.append(adjusted_rand_score(file_tags,final_tags))
        final_tags=[]
        clusters=np.asarray(clusters)
        c=0
        plt.subplots()
        for cluster in clusters:
            cluster=np.asarray(cluster)
            cluster=cluster.astype(np.float)
            plt.scatter(cluster[:,0],cluster[:,1],c=colors[c])
            c+=1
        plt.title("kmeans_"+folders[i])
        plt.savefig("kmeans_"+folders[i]+".png")
    return results

def main():
    path = 'C:/Users/Luis/Desktop/Inteligencia de negocios - Proyecto Clustering/Clustering_Datasets'#Poner dirección de la carpeta donde están los data sets en la variable path
    colors=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
            '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
            '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
            '#CD5C5C','#F08080','#FA8072','#E9967A','#FFA07A',
            '#008080','#000080','#FF00FF','#800080','#800000',
            '#C0C0C0','#808000']
    resultados=[]
    files=os.listdir(path)#Se crea una lista con los nombres de todos los archivos dentro de la ruta definida (variable path)
    labels=list(files)
    sorted_results=[]
    for i in range(0,len(labels)):
        sorted_results.append(list())
    resultados.append(labels)
    resultados.append(kmeans(path,colors))
    resultados.append(average_linkage(path,colors))
    resultados.append(single_linkage(path,colors))
    resultados.append(complete_linkage(path,colors))
    resultados=np.asarray(resultados)

    for i in range(0,len(labels)):
        if(i==0):
            collabel=("Problem name","K-means", "Average Linkage", "Single Linkage","Complete Linkage")
            sorted_results[i]=collabel
        else:
            sorted_results[i]=list(resultados[:,i])

    fig,ax=plt.subplots(figsize=(9,4))
    collabel=("Problem name","K-means", "Average Linkage", "Single Linkage","Complete Linkage")
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=sorted_results,loc='center',fontsize=25)
    fig.savefig("results_table.png")

    sorted_results=np.asarray(sorted_results)
    np.savetxt("output.csv",sorted_results.T,fmt="%s",delimiter=",")

    data=[list(resultados[1]),list(resultados[2]),list(resultados[3]),list(resultados[4])]
    data=np.asarray(data)

    fig,ax=plt.subplots(figsize=(9,4))
    data=data.astype(np.float)
    data=list(data)
    bplot1=ax.boxplot(data,vert=True,patch_artist=True)
    ax.set_xticklabels(["K-means", "Average Linkage", "Single Linkage","Complete Linkage"])
    ax.set_ylabel('ARI')
    fig.savefig("boxplot.png")

main()