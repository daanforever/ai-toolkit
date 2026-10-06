import { PrismaClient } from '@prisma/client';
import { defaultDatasetsFolder, defaultDataRoot } from '@/paths';
import { defaultTrainFolder } from '@/paths';
import NodeCache from 'node-cache';

const myCache = new NodeCache();
const prisma = new PrismaClient();

export const flushCache = () => {
  myCache.flushAll();
};

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) {
    return datasetsPath;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: 'DATASETS_FOLDER',
    },
  });
  datasetsPath = defaultDatasetsFolder;
  if (row?.value && row.value !== '') {
    datasetsPath = row.value;
  }
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) {
    return trainingRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  myCache.set(key, token);
  return token;
};

export const getOpenAIApiUrl = async () => {
  const key = 'OPENAI_API_URL';
  let apiUrl = myCache.get(key) as string;
  if (apiUrl) {
    return apiUrl;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  apiUrl = '';
  if (row?.value && row.value !== '') {
    apiUrl = row.value;
  }
  myCache.set(key, apiUrl);
  return apiUrl;
};

export const getOpenAIDefaultModel = async () => {
  const key = 'OPENAI_DEFAULT_MODEL';
  let model = myCache.get(key) as string;
  if (model) {
    return model;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  model = '';
  if (row?.value && row.value !== '') {
    model = row.value;
  }
  myCache.set(key, model);
  return model;
};

export const getDataRoot = async () => {
  const key = 'DATA_ROOT';
  let dataRoot = myCache.get(key) as string;
  if (dataRoot) {
    return dataRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  dataRoot = defaultDataRoot;
  if (row?.value && row.value !== '') {
    dataRoot = row.value;
  }
  myCache.set(key, dataRoot);
  return dataRoot;
};
