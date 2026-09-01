"""Algoritmo Genético para otimização de hiperparâmetros (Etapa 2).

A representação de um indivíduo é um cromossomo de valores reais normalizados
em [0, 1]; cada gene corresponde a um hiperparâmetro do modelo (ver
``src.models.search_space``), e ``decode_genes`` converte o cromossomo de
volta para parâmetros válidos do scikit-learn.
"""
