import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> }
) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let body: { lr?: number; min_lr?: number; gaussian_mean?: number; gaussian_std?: number; gaussian_mean_2?: number; gaussian_std_2?: number; fixed_cycle_timesteps?: number[]; fixed_cycle_seed?: number | null; fixed_cycle_weight_peak_timesteps?: number[] | null; fixed_cycle_weight_sigma?: number; weight_decay?: number; weight_decay_increment?: number; weight_decay_mode?: string; beta1?: number | null; beta2?: number; content_or_style?: string; timestep_type?: string; network_weights?: number[]; batch_size?: number; gradient_accumulation?: number; save_every?: number; sample_every?: number; warmup_steps?: number; min_snr_gamma?: number; debug?: boolean };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400 }
    );
  }

  const CONTENT_OR_STYLE_VALUES = ['balanced', 'content', 'style', 'gaussian', 'gaussian_bimodal', 'fixed_cycle'] as const;
  const TIMESTEP_TYPE_VALUES = ['sigmoid', 'linear', 'shift', 'flux_shift'] as const;
  const WEIGHT_DECAY_MODE_VALUES = ['update_rms', 'param_rms', 'absolute'] as const;

  const data: { runtime_lr?: number; runtime_min_lr?: number; runtime_gaussian_mean?: number; runtime_gaussian_std?: number; runtime_gaussian_mean_2?: number; runtime_gaussian_std_2?: number; runtime_fixed_cycle_timesteps?: string; runtime_fixed_cycle_seed?: number | null; runtime_fixed_cycle_weight_peak_timesteps?: string | null; runtime_fixed_cycle_weight_sigma?: number; runtime_weight_decay?: number; runtime_weight_decay_increment?: number; runtime_weight_decay_mode?: string; runtime_beta1?: number | null; runtime_beta2?: number; runtime_content_or_style?: string; runtime_timestep_type?: string; runtime_network_weights?: string; runtime_batch_size?: number; runtime_gradient_accumulation?: number; runtime_save_every?: number; runtime_sample_every?: number; runtime_warmup_steps?: number; runtime_min_snr_gamma?: number; runtime_debug?: boolean } = {};

  const lr = body.lr;
  if (lr !== undefined) {
    if (typeof lr !== 'number' || !Number.isFinite(lr)) {
      return NextResponse.json(
        { error: 'lr must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_lr = lr;
  }

  const minLr = body.min_lr;
  if (minLr !== undefined) {
    if (typeof minLr !== 'number' || !Number.isFinite(minLr)) {
      return NextResponse.json(
        { error: 'min_lr must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_min_lr = minLr;
  }

  const gaussianMean = body.gaussian_mean;
  if (gaussianMean !== undefined) {
    if (typeof gaussianMean !== 'number' || !Number.isFinite(gaussianMean) || gaussianMean < 0 || gaussianMean > 999) {
      return NextResponse.json(
        { error: 'gaussian_mean must be a number in [0, 999]' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_mean = gaussianMean;
  }

  const gaussianStd = body.gaussian_std;
  if (gaussianStd !== undefined) {
    if (typeof gaussianStd !== 'number' || !Number.isFinite(gaussianStd)) {
      return NextResponse.json(
        { error: 'gaussian_std must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_std = gaussianStd;
  }

  const gaussianMean2 = body.gaussian_mean_2;
  if (gaussianMean2 !== undefined) {
    if (typeof gaussianMean2 !== 'number' || !Number.isFinite(gaussianMean2) || gaussianMean2 < 0 || gaussianMean2 > 999) {
      return NextResponse.json(
        { error: 'gaussian_mean_2 must be a number in [0, 999]' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_mean_2 = gaussianMean2;
  }

  const gaussianStd2 = body.gaussian_std_2;
  if (gaussianStd2 !== undefined) {
    if (typeof gaussianStd2 !== 'number' || !Number.isFinite(gaussianStd2)) {
      return NextResponse.json(
        { error: 'gaussian_std_2 must be a finite number' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_std_2 = gaussianStd2;
  }

  const fixedCycleTimesteps = body.fixed_cycle_timesteps;
  if (fixedCycleTimesteps !== undefined) {
    if (!Array.isArray(fixedCycleTimesteps) || fixedCycleTimesteps.length === 0) {
      return NextResponse.json(
        { error: 'fixed_cycle_timesteps must be a non-empty array of numbers' },
        { status: 400 }
      );
    }
    for (let i = 0; i < fixedCycleTimesteps.length; i++) {
      const v = fixedCycleTimesteps[i];
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1000) {
        return NextResponse.json(
          { error: `fixed_cycle_timesteps[${i}] must be a finite number in [0, 1000]` },
          { status: 400 }
        );
      }
    }
    data.runtime_fixed_cycle_timesteps = JSON.stringify(fixedCycleTimesteps);
  }

  const fixedCycleSeed = body.fixed_cycle_seed;
  if (fixedCycleSeed !== undefined) {
    if (fixedCycleSeed === null) {
      data.runtime_fixed_cycle_seed = null;
    } else {
      if (typeof fixedCycleSeed !== 'number' || !Number.isFinite(fixedCycleSeed) || !Number.isInteger(fixedCycleSeed) || fixedCycleSeed < 0) {
        return NextResponse.json(
          { error: 'fixed_cycle_seed must be an integer in [0, +inf) or null' },
          { status: 400 }
        );
      }
      data.runtime_fixed_cycle_seed = fixedCycleSeed;
    }
  }

  const fixedCycleWeightPeakTimesteps = body.fixed_cycle_weight_peak_timesteps;
  if (fixedCycleWeightPeakTimesteps !== undefined) {
    if (fixedCycleWeightPeakTimesteps === null) {
      data.runtime_fixed_cycle_weight_peak_timesteps = null;
    } else {
      if (!Array.isArray(fixedCycleWeightPeakTimesteps)) {
        return NextResponse.json(
          { error: 'fixed_cycle_weight_peak_timesteps must be an array of numbers (or null)' },
          { status: 400 }
        );
      }
      for (let i = 0; i < fixedCycleWeightPeakTimesteps.length; i++) {
        const v = fixedCycleWeightPeakTimesteps[i];
        if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1000) {
          return NextResponse.json(
            { error: `fixed_cycle_weight_peak_timesteps[${i}] must be a finite number in [0, 1000]` },
            { status: 400 }
          );
        }
      }
      data.runtime_fixed_cycle_weight_peak_timesteps = JSON.stringify(fixedCycleWeightPeakTimesteps);
    }
  }

  const fixedCycleWeightSigma = body.fixed_cycle_weight_sigma;
  if (fixedCycleWeightSigma !== undefined) {
    if (typeof fixedCycleWeightSigma !== 'number' || !Number.isFinite(fixedCycleWeightSigma) || fixedCycleWeightSigma <= 0) {
      return NextResponse.json(
        { error: 'fixed_cycle_weight_sigma must be a positive finite number' },
        { status: 400 }
      );
    }
    data.runtime_fixed_cycle_weight_sigma = fixedCycleWeightSigma;
  }

  const weightDecay = body.weight_decay;
  if (weightDecay !== undefined) {
    if (typeof weightDecay !== 'number' || !Number.isFinite(weightDecay) || weightDecay < 0) {
      return NextResponse.json(
        { error: 'weight_decay must be a non-negative number' },
        { status: 400 }
      );
    }
    data.runtime_weight_decay = weightDecay;
  }

  const weightDecayIncrement = body.weight_decay_increment;
  if (weightDecayIncrement !== undefined) {
    if (typeof weightDecayIncrement !== 'number' || !Number.isFinite(weightDecayIncrement) || weightDecayIncrement < 0) {
      return NextResponse.json(
        { error: 'weight_decay_increment must be a non-negative number' },
        { status: 400 }
      );
    }
    data.runtime_weight_decay_increment = weightDecayIncrement;
  }

  const weightDecayMode = body.weight_decay_mode;
  if (weightDecayMode !== undefined) {
    if (typeof weightDecayMode !== 'string' || !WEIGHT_DECAY_MODE_VALUES.includes(weightDecayMode)) {
      return NextResponse.json(
        { error: 'weight_decay_mode must be one of: update_rms, param_rms, absolute' },
        { status: 400 }
      );
    }
    data.runtime_weight_decay_mode = weightDecayMode;
  }

  const beta1 = body.beta1;
  if (beta1 !== undefined) {
    if (beta1 === null || beta1 === 0) {
      data.runtime_beta1 = null;
    } else if (typeof beta1 !== 'number' || !Number.isFinite(beta1) || beta1 < 0 || beta1 >= 1) {
      return NextResponse.json(
        { error: 'beta1 must be null or a number in [0, 1)' },
        { status: 400 }
      );
    } else {
      data.runtime_beta1 = beta1;
    }
  }

  const beta2 = body.beta2;
  if (beta2 !== undefined) {
    if (typeof beta2 !== 'number' || !Number.isFinite(beta2) || beta2 <= 0 || beta2 >= 1) {
      return NextResponse.json(
        { error: 'beta2 must be a number in (0, 1)' },
        { status: 400 }
      );
    }
    data.runtime_beta2 = beta2;
  }

  const contentOrStyle = body.content_or_style;
  if (contentOrStyle !== undefined) {
    if (typeof contentOrStyle !== 'string' || !CONTENT_OR_STYLE_VALUES.includes(contentOrStyle)) {
      return NextResponse.json(
        { error: 'content_or_style must be one of: balanced, content, style, gaussian, gaussian_bimodal, fixed_cycle' },
        { status: 400 }
      );
    }
    data.runtime_content_or_style = contentOrStyle;
  }

  const timestepType = body.timestep_type;
  if (timestepType !== undefined) {
    if (typeof timestepType !== 'string' || !TIMESTEP_TYPE_VALUES.includes(timestepType)) {
      return NextResponse.json(
        { error: 'timestep_type must be one of: sigmoid, linear, shift, flux_shift' },
        { status: 400 }
      );
    }
    data.runtime_timestep_type = timestepType;
  }

  const networkWeights = body.network_weights;
  if (networkWeights !== undefined) {
    if (!Array.isArray(networkWeights) || networkWeights.length === 0) {
      return NextResponse.json(
        { error: 'network_weights must be a non-empty array' },
        { status: 400 }
      );
    }
    for (let i = 0; i < networkWeights.length; i++) {
      const w = networkWeights[i];
      if (typeof w !== 'number' || !Number.isFinite(w)) {
        return NextResponse.json(
          { error: `network_weights[${i}] must be a positive number` },
          { status: 400 }
        );
      }
    }
    data.runtime_network_weights = JSON.stringify(networkWeights);
  }

  const batchSize = body.batch_size;
  if (batchSize !== undefined) {
    if (typeof batchSize !== 'number' || !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 128) {
      return NextResponse.json(
        { error: 'batch_size must be an integer between 1 and 128' },
        { status: 400 }
      );
    }
    data.runtime_batch_size = batchSize;
  }

  const gradientAccumulation = body.gradient_accumulation;
  if (gradientAccumulation !== undefined) {
    if (typeof gradientAccumulation !== 'number' || !Number.isInteger(gradientAccumulation) || gradientAccumulation < 1 || gradientAccumulation > 64) {
      return NextResponse.json(
        { error: 'gradient_accumulation must be an integer between 1 and 64' },
        { status: 400 }
      );
    }
    data.runtime_gradient_accumulation = gradientAccumulation;
  }

  const saveEvery = body.save_every;
  if (saveEvery !== undefined) {
    if (typeof saveEvery !== 'number' || !Number.isInteger(saveEvery) || saveEvery < 1 || saveEvery > 10000) {
      return NextResponse.json(
        { error: 'save_every must be an integer between 1 and 10000' },
        { status: 400 }
      );
    }
    data.runtime_save_every = saveEvery;
  }

  const sampleEvery = body.sample_every;
  if (sampleEvery !== undefined) {
    if (typeof sampleEvery !== 'number' || !Number.isInteger(sampleEvery) || sampleEvery < 1 || sampleEvery > 10000) {
      return NextResponse.json(
        { error: 'sample_every must be an integer between 1 and 10000' },
        { status: 400 }
      );
    }
    data.runtime_sample_every = sampleEvery;
  }

  const warmupSteps = body.warmup_steps;
  if (warmupSteps !== undefined) {
    if (typeof warmupSteps !== 'number' || !Number.isInteger(warmupSteps) || warmupSteps < 0 || warmupSteps > 100000) {
      return NextResponse.json(
        { error: 'warmup_steps must be an integer between 0 and 100000' },
        { status: 400 }
      );
    }
    data.runtime_warmup_steps = warmupSteps;
  }

  const minSnrGamma = body.min_snr_gamma;
  if (minSnrGamma !== undefined) {
    if (typeof minSnrGamma !== 'number' || !Number.isFinite(minSnrGamma) || minSnrGamma < 0 || minSnrGamma > 100) {
      return NextResponse.json(
        { error: 'min_snr_gamma must be a number between 0 and 100' },
        { status: 400 }
      );
    }
    data.runtime_min_snr_gamma = minSnrGamma;
  }

  const debug = body.debug;
  if (debug !== undefined) {
    if (typeof debug !== 'boolean') {
      return NextResponse.json(
        { error: 'debug must be a boolean' },
        { status: 400 }
      );
    }
    data.runtime_debug = debug;
  }

  if (Object.keys(data).length === 0) {
    return NextResponse.json(
      { error: 'At least one of lr, min_lr, gaussian_mean, gaussian_std, gaussian_mean_2, gaussian_std_2, fixed_cycle_timesteps, fixed_cycle_seed, fixed_cycle_weight_peak_timesteps, fixed_cycle_weight_sigma, weight_decay, weight_decay_increment, weight_decay_mode, beta1, beta2, content_or_style, timestep_type, network_weights, batch_size, gradient_accumulation, save_every, sample_every, warmup_steps, min_snr_gamma, debug must be provided' },
      { status: 400 }
    );
  }

  const updated = await prisma.runtimeParams.upsert({
    where: { jobId: jobID },
    update: data,
    create: { jobId: jobID, ...data },
  });

  return NextResponse.json(updated);
}
