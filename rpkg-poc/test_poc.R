source("rpkg-poc/binary_backend.R")
d <- read.csv("testdata/sim.csv")
m <- as.matrix(d)
cat("input matrix:", nrow(m), "x", ncol(m), "\n\n")

t0 <- Sys.time()
res <- hmm_inference_bin(m, method = "map")
cat("MAP elapsed:", round(as.numeric(difftime(Sys.time(), t0, units="secs")), 2), "s\n")
cat("Se (fixed):", round(res$Se, 3), "\n")
print(head(res$prevalence, 3))

t0 <- Sys.time()
res2 <- hmm_inference_bin(m, method = "map", infer_sesp = TRUE)
cat("\nMAP + infer_sesp:", round(as.numeric(difftime(Sys.time(), t0, units="secs")), 2), "s\n")
cat("Se est :", round(res2$Se, 3), "\n")
cat("Se true: 0.407 0.407 0.100 0.650 0.809 0.492\n")

t0 <- Sys.time()
res3 <- hmm_inference_bin(m, method = "nuts", nuts_samples = 500, warmup = 500)
cat("\nNUTS 500+500:", round(as.numeric(difftime(Sys.time(), t0, units="secs")), 2), "s\n")
cat("draws:", nrow(res3$draws), "x", ncol(res3$draws), "\n")
cat("posterior sd(pi1_0):", round(sd(res3$draws$pi1_0), 4), "\n")

res4 <- hmm_inference_bin(m, method = "map", tests = 4)
cat("\nculture-only (tests=4) prevalence at t=10:",
    round(res4$prevalence$prevalence[res4$prevalence$time == 10], 4), "\n")
cat("\nALL MODES OK\n")
