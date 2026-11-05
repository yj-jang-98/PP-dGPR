// Multiparty CKKS aggregation backend for the HE-PoE experiment.
package main

import (
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type request struct {
	Values  [][]float64 `json:"values"`
	Workers int         `json:"workers"`
}

type response struct {
	Values           []float64 `json:"values"`
	Parties          int       `json:"parties"`
	Slots            int       `json:"slots"`
	Ciphertexts      int       `json:"ciphertexts"`
	LogN             int       `json:"log_n"`
	LogQ             int       `json:"log_q"`
	LogScale         int       `json:"log_scale"`
	SecretH          int       `json:"secret_h"`
	EphemeralRho     float64   `json:"ephemeral_rho"`
	ErrorSigma       float64   `json:"error_sigma"`
	AuxiliaryModuli  int       `json:"auxiliary_moduli"`
	Workers          int       `json:"workers"`
	KeyGenerationMS  float64   `json:"key_generation_ms"`
	SecretKeyMS      float64   `json:"secret_key_generation_ms"`
	PublicShareMS    float64   `json:"public_key_share_ms"`
	PublicCombineMS  float64   `json:"public_key_aggregation_ms"`
	EncryptionMS     float64   `json:"encryption_ms"`
	AggregationMS    float64   `json:"aggregation_ms"`
	DecryptionMS     float64   `json:"decryption_ms"`
	DecryptShareMS   float64   `json:"decryption_share_ms"`
	DecryptCombineMS float64   `json:"decryption_aggregation_ms"`
	DecodeMS         float64   `json:"decode_ms"`
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func parallelFor(n, workers int, operation func(int) error) error {
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	jobs := make(chan int)
	errs := make(chan error, n)
	var wait sync.WaitGroup
	wait.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer wait.Done()
			for index := range jobs {
				if err := operation(index); err != nil {
					errs <- err
				}
			}
		}()
	}
	for index := 0; index < n; index++ {
		jobs <- index
	}
	close(jobs)
	wait.Wait()
	close(errs)
	for err := range errs {
		return err
	}
	return nil
}

func main() {
	var req request
	check(json.NewDecoder(os.Stdin).Decode(&req))
	if len(req.Values) == 0 || len(req.Values[0]) == 0 {
		panic("values must be a non-empty parties-by-slots matrix")
	}
	parties := len(req.Values)
	workers := req.Workers
	if workers < 1 {
		workers = 1
	}
	if workers > parties {
		workers = parties
	}
	slots := len(req.Values[0])
	for i := range req.Values {
		if len(req.Values[i]) != slots {
			panic("all parties must provide the same number of slots")
		}
	}

	// Nawaz et al.: N=2^16 and an approximately 800-bit ciphertext modulus.
	// 55 + 15*45 + 2*35 = 800 bits. Only addition is evaluated, so no
	// relinearization or Galois keys are needed.
	logQ := []int{55}
	for i := 0; i < 15; i++ {
		logQ = append(logQ, 45)
	}
	logQ = append(logQ, 35, 35)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,
		LogQ:            logQ,
		LogDefaultScale: 45,
		Xs:              ring.Ternary{P: 0.5},
		Xe:              ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2},
	})
	check(err)
	maxSlots := params.MaxSlots()
	chunks := (slots + maxSlots - 1) / maxSlots

	keyStart := time.Now()
	secretPointers := make([]*rlwe.SecretKey, parties)
	stageStart := time.Now()
	check(parallelFor(parties, workers, func(i int) error {
		// Definition 6 specifies a secret of exact Hamming weight h=64.
		secretPointers[i] = rlwe.NewKeyGenerator(params).GenSecretKeyWithHammingWeightNew(64)
		return nil
	}))
	secretKeyDuration := time.Since(stageStart)

	crs, err := sampling.NewKeyedPRNG([]byte("Nawaz-SDS-GP-Lattigo-v6.2"))
	check(err)
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crp := ckg.SampleCRP(crs)
	pkShares := make([]multiparty.PublicKeyGenShare, parties)
	stageStart = time.Now()
	check(parallelFor(parties, workers, func(i int) error {
		localCKG := multiparty.NewPublicKeyGenProtocol(params)
		pkShares[i] = localCKG.AllocateShare()
		localCKG.GenShare(secretPointers[i], crp, &pkShares[i])
		return nil
	}))
	publicShareDuration := time.Since(stageStart)
	combinedPKShare := ckg.AllocateShare()
	stageStart = time.Now()
	for i := range pkShares {
		ckg.AggregateShares(pkShares[i], combinedPKShare, &combinedPKShare)
	}
	collectivePK := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combinedPKShare, crp, collectivePK)
	publicCombineDuration := time.Since(stageStart)
	keyDuration := time.Since(keyStart)

	evaluator := ckks.NewEvaluator(params, nil)
	ciphertexts := make([][]*rlwe.Ciphertext, parties)
	encryptionStart := time.Now()
	check(parallelFor(parties, workers, func(i int) error {
		encoder := ckks.NewEncoder(params)
		encryptor := rlwe.NewEncryptor(params, collectivePK)
		ciphertexts[i] = make([]*rlwe.Ciphertext, chunks)
		for chunk := 0; chunk < chunks; chunk++ {
			begin := chunk * maxSlots
			end := min(begin+maxSlots, slots)
			pt := ckks.NewPlaintext(params, params.MaxLevel())
			if err := encoder.Encode(req.Values[i][begin:end], pt); err != nil {
				return err
			}
			ct, err := encryptor.EncryptNew(pt)
			if err != nil {
				return err
			}
			ciphertexts[i][chunk] = ct
		}
		return nil
	}))
	encryptionDuration := time.Since(encryptionStart)
	aggregationStart := time.Now()
	aggregates := make([]*rlwe.Ciphertext, chunks)
	for chunk := 0; chunk < chunks; chunk++ {
		aggregates[chunk] = ciphertexts[0][chunk].CopyNew()
		for i := 1; i < parties; i++ {
			check(evaluator.Add(aggregates[chunk], ciphertexts[i][chunk], aggregates[chunk]))
		}
	}
	aggregationDuration := time.Since(aggregationStart)

	decryptionStart := time.Now()
	// N-out-of-N collective key switching to the zero key is collective
	// decryption: no individual party ever possesses the aggregate secret key.
	// Nawaz et al. use sigma=3.2 for the Gaussian error distribution.
	noise := ring.DiscreteGaussian{Sigma: 3.2, Bound: 20}
	cks, err := multiparty.NewKeySwitchProtocol(params, noise)
	check(err)
	cksShares := make([][]multiparty.KeySwitchShare, chunks)
	shareStart := time.Now()
	for chunk := 0; chunk < chunks; chunk++ {
		cksShares[chunk] = make([]multiparty.KeySwitchShare, parties)
		chunkIndex := chunk
		check(parallelFor(parties, workers, func(i int) error {
			localCKS, err := multiparty.NewKeySwitchProtocol(params, noise)
			if err != nil {
				return err
			}
			cksShares[chunkIndex][i] = localCKS.AllocateShare(aggregates[chunkIndex].Level())
			zero := rlwe.NewSecretKey(params)
			localCKS.GenShare(
				secretPointers[i], zero, aggregates[chunkIndex],
				&cksShares[chunkIndex][i],
			)
			return nil
		}))
	}
	decryptShareDuration := time.Since(shareStart)
	combineStart := time.Now()
	switched := make([]*rlwe.Ciphertext, chunks)
	for chunk := 0; chunk < chunks; chunk++ {
		combinedShare := cks.AllocateShare(aggregates[chunk].Level())
		for i := range cksShares[chunk] {
			check(cks.AggregateShares(cksShares[chunk][i], combinedShare, &combinedShare))
		}
		switched[chunk] = ckks.NewCiphertext(params, 1, aggregates[chunk].Level())
		cks.KeySwitch(aggregates[chunk], combinedShare, switched[chunk])
	}
	decryptCombineDuration := time.Since(combineStart)
	decodeStart := time.Now()
	zero := rlwe.NewSecretKey(params)
	decoded := make([]float64, slots)
	decryptor := rlwe.NewDecryptor(params, zero)
	decoder := ckks.NewEncoder(params)
	for chunk := 0; chunk < chunks; chunk++ {
		begin := chunk * maxSlots
		end := min(begin+maxSlots, slots)
		plaintext := decryptor.DecryptNew(switched[chunk])
		check(decoder.Decode(plaintext, decoded[begin:end]))
	}
	decodeDuration := time.Since(decodeStart)
	decryptionDuration := time.Since(decryptionStart)

	check(json.NewEncoder(os.Stdout).Encode(response{
		Values:           decoded,
		Parties:          parties,
		Slots:            slots,
		Ciphertexts:      chunks,
		LogN:             params.LogN(),
		LogQ:             int(params.LogQP() - params.LogP()),
		LogScale:         int(params.LogDefaultScale()),
		SecretH:          64,
		EphemeralRho:     0.5,
		ErrorSigma:       3.2,
		AuxiliaryModuli:  params.PCount(),
		Workers:          workers,
		KeyGenerationMS:  float64(keyDuration.Microseconds()) / 1000,
		SecretKeyMS:      float64(secretKeyDuration.Microseconds()) / 1000,
		PublicShareMS:    float64(publicShareDuration.Microseconds()) / 1000,
		PublicCombineMS:  float64(publicCombineDuration.Microseconds()) / 1000,
		EncryptionMS:     float64(encryptionDuration.Microseconds()) / 1000,
		AggregationMS:    float64(aggregationDuration.Microseconds()) / 1000,
		DecryptionMS:     float64(decryptionDuration.Microseconds()) / 1000,
		DecryptShareMS:   float64(decryptShareDuration.Microseconds()) / 1000,
		DecryptCombineMS: float64(decryptCombineDuration.Microseconds()) / 1000,
		DecodeMS:         float64(decodeDuration.Microseconds()) / 1000,
	}))
}
