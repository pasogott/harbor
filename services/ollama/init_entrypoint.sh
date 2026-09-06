#!/bin/sh

echo "Harbor: ollama init"

warn_pull_failure() {
  echo "WARNING: Failed to pull model '$1'. Continuing with remaining models..."
  if [ -n "$2" ]; then
    echo "$2"
  fi
}

main() {
  pull_default_models
  # Marker is read by the healthcheck; tail keeps the sidecar in running|healthy
  # so `compose --wait` doesn't flag a clean exit as premature failure.
  mkdir -p /run/harbor && touch /run/harbor/ollama-init-done
  exec tail -f /dev/null
}

pull_default_models() {
  echo "Pulling default models:"
  echo "${HARBOR_OLLAMA_DEFAULT_MODELS:-}"

  if [ -z "${HARBOR_OLLAMA_DEFAULT_MODELS:-}" ]; then
    echo "No default models to pull"
    return
  fi

  host=${OLLAMA_HOST:-http://ollama:11434}
  models=$HARBOR_OLLAMA_DEFAULT_MODELS
  while [ -n "$models" ]; do
    model=${models%%,*}
    if [ "$models" = "$model" ]; then
      models=
    else
      models=${models#*,}
    fi

    model=$(printf '%s' "$model" | tr -d '[:space:]')
    if [ -z "$model" ]; then
      continue
    fi

    echo "Pulling model $model"
    payload=$(printf '{"model":"%s","stream":false}' "$model")
    if response=$(wget -qO- \
        --header 'Content-Type: application/json' \
        --post-data "$payload" \
        "$host/api/pull" 2>&1); then
      if ! printf '%s' "$response" | grep -q '"status":"success"'; then
        warn_pull_failure "$model" "$response"
      fi
    else
      warn_pull_failure "$model" "$response"
    fi
  done
}

main
