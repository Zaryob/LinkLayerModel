# LinkLayerModel

A historical Rust rewrite of the code accompanying Marco Zuniga and Bhaskar Krishnamachari's “[Link Layer Models for Wireless Sensor Networks](http://anrg.usc.edu/www/download_files/LinkModellingTutorial.pdf)”. The model and original research retain their authors' attribution. See [LICENCE](LICENCE) for the retained license.

## Run from source

Install Rust/Cargo, clone this repository and work from its root:

```sh
git clone https://github.com/Zaryob/LinkLayerModel.git
cd LinkLayerModel
cargo build
cargo run
```

Edit `topology.conf`, a TOML configuration file, before running. The executable reads it from the **current directory**; it has no configuration-path command-line argument. The topology selector is `top` (not `topology`): 1 grid, 2 uniform, 3 random, 4 file. The supplied example uses `top = 4` and `top_file = 'topologyFile'` with 40 nodes.

For file topology, retain the supplied `topology = [` / `];` format and one `node_id X Y` row per node; IDs must be zero-based and within `num_nodes`. Coordinates and distances are in meters. Run only with trusted local configuration/data: comprehensive malformed-input handling has not been established.

The program writes **`outputFile` in the current directory**, replacing an existing file of that name. Run in a copied example directory if you want to retain a previous result. The output contains node placement, packet reception rate, received power and RSSI matrices.

## Validation and limits

On 9 October 2026, source baseline `18fa194280dcc67598ac8080d31e5522b1fa05e3` built with rustc/cargo 1.91.1 on macOS 27.0 arm64. Running the supplied file-topology example from an isolated working directory exited successfully and produced output, but the output contained **40 non-finite (`NaN`) entries**. A successful exit does not establish numerical correctness. The code uses unseeded random samples, so it does not promise identical numerical output on repeat runs.

This is historical/research code, with no verified release binary, automated model-correctness suite or independently established agreement with the paper. [Numerical validation and source-run backlog](https://github.com/Zaryob/LinkLayerModel/issues/2) tracks the remaining work. Do not use the smoke result as evidence of a physically valid radio simulation.
