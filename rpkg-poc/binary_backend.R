# Proof of concept: drive the trimmed test2inf binary from R.
#
# Replaces the JuliaCall bridge. No Julia on the user's machine, no
# instantiation, no first-call compilation.

test2inf_bin <- function() {
  exe <- Sys.getenv("TEST2INF_BIN", "")
  if (nzchar(exe) && file.exists(exe)) return(exe)
  # In a real package: system.file("bin", "test2inf.exe", package = "test2infeR")
  cand <- file.path(getwd(), "dist", "bin",
                    if (.Platform$OS.type == "windows") "test2inf.exe" else "test2inf")
  if (file.exists(cand)) return(cand)
  stop("test2inf binary not found; set TEST2INF_BIN")
}

hmm_inference_bin <- function(test_mat,
                              method = c("map", "nuts"),
                              tests = NULL,
                              infer_sesp = FALSE,
                              penalty = TRUE,
                              seasons = 4L,
                              repeat_captures = c("stack", "pool", "last"),
                              nuts_samples = 1000L,
                              warmup = 1000L,
                              target_acc = 0.8,
                              seed = 1L,
                              workdir = tempfile("test2inf")) {
  method <- match.arg(method)
  repeat_captures <- match.arg(repeat_captures)

  dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(workdir, recursive = TRUE), add = TRUE)

  # The binary takes a CSV, so write the matrix out with the fixed schema.
  stopifnot(is.matrix(test_mat), ncol(test_mat) >= 9)
  df <- as.data.frame(test_mat[, 1:9, drop = FALSE])
  names(df) <- c("time", "id", "captured", paste0("test", 1:6))
  csv <- file.path(workdir, "data.csv")
  utils::write.csv(df, csv, row.names = FALSE, na = "NA")

  outdir <- file.path(workdir, "out")
  args <- c("--data", csv, "--out", outdir, "--method", method,
            "--seasons", seasons, "--repeat", repeat_captures, "--seed", seed)
  if (!is.null(tests))  args <- c(args, "--tests", paste(tests, collapse = ","))
  if (isTRUE(infer_sesp)) args <- c(args, "--infer-sesp")
  if (!isTRUE(penalty))   args <- c(args, "--no-penalty")
  if (method == "nuts") {
    args <- c(args, "--draws", nuts_samples, "--warmup", warmup,
              "--accept", target_acc)
  }

  res <- system2(test2inf_bin(), args = as.character(args),
                 stdout = TRUE, stderr = TRUE)
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop("test2inf failed:\n", paste(res, collapse = "\n"))
  }

  rd <- function(f) {
    p <- file.path(outdir, f)
    if (file.exists(p)) utils::read.csv(p, stringsAsFactors = FALSE) else NULL
  }

  params <- rd("parameters.csv")
  # MAP writes an `estimate` column; NUTS writes `mean` and `sd`.
  valcol <- if ("estimate" %in% names(params)) "estimate" else "mean"
  getp <- function(pat) {
    i <- grep(pat, params$parameter)
    if (length(i) == 0L) return(numeric(0))
    stats::setNames(params[[valcol]][i], params$parameter[i])
  }

  list(
    log        = res,
    parameters = params,
    prevalence = rd("prevalence.csv"),
    p_infected = rd("p_infected.csv"),
    year_effects = rd("year_effects.csv"),
    draws      = rd("draws.csv"),
    Se         = unname(getp("^Se[[]")),
    Sp         = unname(getp("^Sp[[]")),
    settings   = list(method = method, tests = tests, infer_sesp = infer_sesp,
                      penalty = penalty, seasons = seasons,
                      repeat_captures = repeat_captures, seed = seed)
  )
}
