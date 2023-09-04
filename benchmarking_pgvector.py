if __name__ ==  '__main__':

    import sys
    sys.path.append('/Users/eshao1/vector_databases/vector-db-benchmark') 

    import stopit
    import typer
    import fnmatch
    import traceback

    from benchmark.config_read import read_dataset_config, read_engine_configs
    from engine.clients.client_factory import ClientFactory
    from benchmark.dataset import Dataset
    from engine.base_client import IncompatibilityError

    all_engines = read_engine_configs()
    pgvector_engines = {key:all_engines[key] for key in all_engines if 'pgvector' in key} 

    all_datasets = read_dataset_config()
    dataset_fashion_mnist = all_datasets["deep-image-96-angular"]

    host = "<host name here>" 
    client = ClientFactory(host).build_client(pgvector_engines["pgvector-default"])

    dataset = Dataset(dataset_fashion_mnist)


    engine_name = "pgvector-default"
    dataset_name = dataset_fashion_mnist["name"]
    exit_on_error = True
    timeout = 86400.0
    skip_upload = False
    skip_search = False

    try:
        with stopit.ThreadingTimeout(timeout) as tt:
            client.run_experiment(dataset, skip_upload, skip_search)

        # If the timeout is reached, the server might be still in the
        # middle of some background processing, like creating the index.
        # Next experiment should not be launched. It's better to reset
        # the server state manually.
        if tt.state != stopit.ThreadingTimeout.EXECUTED:
            print(
                f"Timed out {engine_name} - {dataset_name}, "
                f"exceeded {timeout} seconds"
            )
            exit(2)
    except IncompatibilityError as e:
        print(f"Skipping {engine_name} - {dataset_name}, incompatible params")
        
    except KeyboardInterrupt as e:
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"Experiment {engine_name} - {dataset_name} interrupted")
        traceback.print_exc()
        if exit_on_error:
            raise e
    