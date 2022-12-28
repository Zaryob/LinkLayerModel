use confy;
use serde_derive::{Deserialize, Serialize};

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};
use std::path::Path;

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

#[derive(Default, Serialize, Deserialize)]
struct InputVariables {
    n: f64,
    sigma: f64,
    pld0: f64,
    d0: f64,
    mod_: i32,
    enc: i32,
    pout: f64,
    pn: f64,
    pre: i32,
    fra: i32,
    num_nodes: i32,
    top: i32,
    grid: f64,
    xterr: f64,
    yterr: f64,
    top_file: String,
    area: f64,
    s11: f64,
    s12: f64,
    s21: f64,
    s22: f64,
}

impl InputVariables {
    fn new() -> InputVariables {
        InputVariables {
            n: 0.0,
            sigma: 0.0,
            pld0: 0.0,
            d0: 0.0,
            mod_: 0,
            enc: 0,
            pout: 0.0,
            pn: 0.0,
            pre: 0,
            fra: 0,
            num_nodes: 0,
            top: 0,
            grid: 0.0,
            xterr: 0.0,
            yterr: 0.0,
            top_file: "".to_string(),
            area: 0.0,
            s11: 0.0,
            s12: 0.0,
            s21: 0.0,
            s22: 0.0,
        }
    }
}

struct OutputVariables {
    node_pos_x: Vec<f64>,   // X coordinate
    node_pos_y: Vec<f64>,   // Y coordinate
    output_power: Vec<f64>, // output power
    noise_floor: Vec<f64>,  // noise floor
    gen: Vec<Vec<f64>>,     // general double dimensional array for rssi, Pe and prr
    pr: Vec<Vec<f64>>,      // general double dimensional array for rssi, Pe and prr
}

impl OutputVariables {
    fn new(num_nodes: usize) -> Self {
        OutputVariables {
            node_pos_x: vec![0.0; num_nodes],
            node_pos_y: vec![0.0; num_nodes],
            output_power: vec![0.0; num_nodes],
            noise_floor: vec![0.0; num_nodes],
            gen: vec![vec![0.0; num_nodes]; num_nodes],
            pr: vec![vec![0.0; num_nodes]; num_nodes],
        }
    }
}

fn read_topology_file(
    input_file: &str,
    in_var: &InputVariables,
    out_var: &mut OutputVariables,
) -> Result<bool, Box<dyn Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let mut counter = 0;
    for line in reader.lines() {
        let line = line?;
        if !line.is_empty()
            && !line.starts_with('%')
            && !line.starts_with('t')
            && !line.starts_with(']')
        {
            let mut tokens = line.split_ascii_whitespace();
            let node = usize::from_str(tokens.next().ok_or("Error: Invalid node number")?)?;
            let x = f64::from_str(tokens.next().ok_or("Error: Invalid x coordinate")?)?;
            let y = f64::from_str(tokens.next().ok_or("Error: Invalid y coordinate")?)?;
            out_var.node_pos_x[node] = x;
            out_var.node_pos_y[node] = y;
            counter += 1;
        }
    }
    if counter != in_var.num_nodes {
        return Err(format!(
            "Number of nodes in file {} does not agree with value entered in NUMBER_OF_NODES",
            input_file
        )
        .into());
    }

    Ok(true)
}

fn obtain_topology(in_var: &InputVariables, out_var: &mut OutputVariables) -> bool {
    let mut rand = rand::thread_rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut wrong_placement = true;
    let i = 0;
    let j = 0;
    match in_var.top {
        1 => {
            if in_var.grid < in_var.d0 {
                println!("Error: value of GRID_UNIT must be greater than D0");
                return false;
            }
            let sqrt_num_nodes = (in_var.num_nodes as f64).sqrt().floor() as i32;
            if sqrt_num_nodes as f64 != (in_var.num_nodes as f64).sqrt() {
                println!("Error: on GRID topology, NUMBER_OF_NODES should be the square of a natural number");
                return false;
            }
            for i in 0..in_var.num_nodes {
                out_var.node_pos_x[i as usize] = (i % sqrt_num_nodes) as f64 * in_var.grid as f64;
                out_var.node_pos_y[i as usize] = (i / sqrt_num_nodes) as f64 * in_var.grid as f64;
            }
        }
        2 => {
            if in_var.xterr < 0.0 || in_var.yterr < 0.0 {
                println!("Error: values of TERRAIN_DIMENSIONS must be positive");
                return false;
            }
            let cell_length = ((in_var.area / in_var.num_nodes as f64) as f64).sqrt();
            let nodes_x = (in_var.xterr as f64 / cell_length).ceil() as i32;
            let cell_length = in_var.xterr as f64 / nodes_x as f64;
            if cell_length < in_var.d0 as f64 * 1.4 {
                println!(
                    "Error: on UNIFORM topology, density is too high, increase physical terrain"
                );
                return false;
            }
            for i in 0..in_var.num_nodes {
                out_var.node_pos_x[i as usize] = (i % nodes_x) as f64 * cell_length
                    + normal.sample(&mut rand::thread_rng()) * cell_length;
                out_var.node_pos_y[i as usize] = (i / nodes_x) as f64 * cell_length
                    + normal.sample(&mut rand::thread_rng()) * cell_length;
                while wrong_placement {
                    for j in 0..i {
                        let x_dist =
                            out_var.node_pos_x[i as usize] - out_var.node_pos_x[j as usize];
                        let y_dist =
                            out_var.node_pos_y[i as usize] - out_var.node_pos_y[j as usize];
                        // distance between a given pair of nodes
                        let dist = ((x_dist * x_dist) + (y_dist * y_dist)).sqrt();
                        if dist < in_var.d0 {
                            out_var.node_pos_x[i as usize] = (i % nodes_x) as f64 * cell_length
                                + normal.sample(&mut rand::thread_rng()) * cell_length;
                            out_var.node_pos_y[i as usize] = (i / nodes_x) as f64 * cell_length
                                + normal.sample(&mut rand::thread_rng()) * cell_length;
                            wrong_placement = true;
                            break;
                        }
                    }
                    if j == i {
                        wrong_placement = false;
                    }
                }
            }
        }
        3 => {
            if in_var.xterr < 0.0 || in_var.xterr < 0.0 {
                println!("Error: values of TERRAIN_DIMENSIONS must be positive");
                std::process::exit(1);
            }
            let cell_length = (in_var.area / in_var.num_nodes as f64).sqrt();
            if cell_length < in_var.d0 * 1.4 {
                println!(
                    "Error: on RANDOM topology, density is too high, increase physical terrain"
                );
                std::process::exit(1);
            }
            for i in 0..in_var.num_nodes {
                let mut wrong_placement = true;
                while wrong_placement {
                    out_var.node_pos_x[i as usize] =
                        normal.sample(&mut rand::thread_rng()) * in_var.xterr;
                    out_var.node_pos_y[i as usize] =
                        normal.sample(&mut rand::thread_rng()) * in_var.yterr;
                    wrong_placement = false;
                    for j in 0..i {
                        let x_dist =
                            out_var.node_pos_x[i as usize] - out_var.node_pos_x[j as usize];
                        let y_dist =
                            out_var.node_pos_y[i as usize] - out_var.node_pos_y[j as usize];
                        let dist = (x_dist * x_dist + y_dist * y_dist).sqrt();
                        if dist < in_var.d0 {
                            wrong_placement = true;
                            break;
                        }
                    }
                }
            }
        }
        4 => {
            // create topology
            // readTopologyFile(in_var.topFile, );
            read_topology_file(&in_var.top_file, in_var, out_var);
            correct_topology(in_var, out_var);
        }
        _ => {
            println!("Error: topology is not correct, please check TOPOLOGY in the input file");
            std::process::exit(1);
        }
    }
    return true;
}

fn correct_topology(in_var: &InputVariables, out_var: &OutputVariables) -> bool {
    for i in 0..in_var.num_nodes {
        for j in (i + 1)..in_var.num_nodes {
            let x_dist = out_var.node_pos_x[i as usize] - out_var.node_pos_x[j as usize];
            let y_dist = out_var.node_pos_y[i as usize] - out_var.node_pos_y[j as usize];
            let dist = (x_dist * x_dist + y_dist * y_dist).sqrt();
            if dist < in_var.d0 {
                println!(
                    "Error: file {} contains inter_node distances less than one.",
                    in_var.top_file
                );
                std::process::exit(1);
            }
        }
    }
    true
}

fn obtain_radio_pt_pn(in_var: &InputVariables, out_var: &mut OutputVariables) -> bool {
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let t11 = in_var.s11.sqrt();
    let t12 = in_var.s12 / t11;
    let t22 = (in_var.s11 * in_var.s22 - in_var.s12.powi(2)) / in_var.s11;
    for i in 0..in_var.num_nodes {
        let rn1 = normal.sample(&mut rand::thread_rng());
        let rn2 = normal.sample(&mut rand::thread_rng());
        out_var.noise_floor[i as usize] = in_var.pn + t11 * rn1;
        out_var.output_power[i as usize] = in_var.pout + t12 * rn1 + t22 * rn2;
    }
    true
}

fn obtain_rssi(in_var: &InputVariables, out_var: &mut OutputVariables) -> bool {
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for i in 0..in_var.num_nodes {
        for j in i + 1..in_var.num_nodes {
            let x_dist = out_var.node_pos_x[i as usize] - out_var.node_pos_x[j as usize];
            let y_dist = out_var.node_pos_y[i as usize] - out_var.node_pos_y[j as usize];
            // distance between a given pair of nodes
            let dist = (x_dist * x_dist + y_dist * y_dist).sqrt();
            // mean decay dependent on distance
            let avg_decay = -in_var.pld0
                - 10.0 * in_var.n * (dist / in_var.d0).log2() / std::f64::consts::LOG10_2
                + normal.sample(&mut rand::thread_rng()) * in_var.sigma;
            // assymetric links are given by running two different
            // R.V.s for each unidirectional link.
            // NOTE: this approach is not accurate, assymetry is due mainly to
            //		 to hardware imperfections and not for assymetric paths
            out_var.gen[i as usize][j as usize] = out_var.output_power[i as usize] + avg_decay;
            out_var.gen[j as usize][i as usize] = out_var.output_power[j as usize] + avg_decay;
        }
    }
    for i in 0..in_var.num_nodes {
        for j in 0..in_var.num_nodes {
            let dec = in_var.pld0
                + 10.0 * in_var.n * out_var.node_pos_x[j as usize].log2()
                    / std::f64::consts::LOG10_2
                + normal.sample(&mut rand::thread_rng()) * in_var.sigma;
            out_var.pr[i as usize][j as usize] = -dec;
        }
    }
    true
}

fn obtain_prr(in_var: &InputVariables, out_var: &mut OutputVariables) -> bool {
    for i in 0..in_var.num_nodes {
        for j in 0..in_var.num_nodes {
            if i == j {
                out_var.gen[i as usize][j as usize] = 1.0;
            } else {
                let pre_seq = (1.0 - out_var.gen[i as usize][j as usize]).powi(8 * in_var.pre);
                match in_var.enc {
                    1 => {
                        // NRZ
                        out_var.gen[i as usize][j as usize] = pre_seq
                            * (1.0 - out_var.gen[i as usize][j as usize])
                                .powi(8 * (in_var.fra - in_var.pre));
                    }
                    2 => {
                        // 4B5B
                        out_var.gen[i as usize][j as usize] = pre_seq
                            * (1.0 - out_var.gen[i as usize][j as usize])
                                .powf(8.0 * ((in_var.fra - in_var.pre) as f64) * 1.25);
                    }
                    3 => {
                        // Manchester
                        out_var.gen[i as usize][j as usize] = pre_seq
                            * (1.0 - out_var.gen[i as usize][j as usize])
                                .powi(8 * (in_var.fra - in_var.pre) * 2);
                    }
                    4 => {
                        // SECDED
                        out_var.gen[i as usize][j as usize] = pre_seq
                            * ((1.0 - out_var.gen[i as usize][j as usize]).powi(8)
                                + 8.0
                                    * out_var.gen[i as usize][j as usize]
                                    * (1.0 - out_var.gen[i as usize][j as usize]).powi(7))
                            .powi((in_var.fra - in_var.pre) * 3);
                    }
                    _ => {
                        println!("Error: encoding is not correct, please check ENCODING in the input file");
                        std::process::exit(1);
                    }
                }
            }
        }
    }

    true
}

fn obtain_prob_error(in_var: &InputVariables, out_var: &mut OutputVariables) -> bool {
    for i in 0..in_var.num_nodes {
        for j in 0..in_var.num_nodes {
            if i == j {
                out_var.gen[i as usize][j as usize] = 0.0;
            } else {
                // convert SNR from dBm to Watts
                let snr = (10.0f64.powf(
                    (out_var.gen[i as usize][j as usize] - out_var.noise_floor[j as usize]) / 10.0,
                )) / 0.64;
                // division by 0.64 above (snr) converts from Eb/No to RSSI
                // this is specific for each radio (read paper: Data-rate(R) / Bandwidth-noise(B))
                match in_var.mod_ {
                    1 => {
                        // NCASK
                        out_var.gen[i as usize][j as usize] =
                            0.5 * (snr.exp() * -0.5 + Q(snr.sqrt()));
                    }
                    2 => {
                        // ASK
                        out_var.gen[i as usize][j as usize] = Q((snr / 2.0).sqrt());
                    }
                    3 => {
                        // NCFSK
                        out_var.gen[i as usize][j as usize] = 0.5 * snr.exp() * -0.5;
                    }
                    4 => {
                        // FSK
                        out_var.gen[i as usize][j as usize] = Q(snr.sqrt());
                    }
                    5 => {
                        // BPSK
                        out_var.gen[i as usize][j as usize] = Q((2.0 * snr).sqrt());
                    }
                    6 => {
                        // DPSK
                        out_var.gen[i as usize][j as usize] = 0.5 * snr.exp() * -1.0;
                    }
                    _ => {
                        println!("Error: modulation is not correct, please check MODULATION in the input file");
                        std::process::exit(1);
                    }
                }
            }
        }
    }

    true
}

fn Q(z: f64) -> f64 {
    let a1 = 0.127414796;
    let a2 = -0.142248368;
    let a3 = 0.7107068705;
    let a4 = -0.7265760135;
    let a5 = 0.5307027145;
    let b = 0.231641888;
    let t = 1.0 / (1.0 + b * z);

    if z >= 0.0 {
        (a1 * t + a2 * t.powi(2) + a3 * t.powi(3) + a4 * t.powi(4) + a5 * t.powi(5))
            * (-z.powi(2) / 2.0).exp()
    } else {
        println!("Error in Q function: argument Z must be greater equal than 0");
        std::process::exit(1);
    }
}

fn print_file(
    output_file: &str,
    in_var: &InputVariables,
    out_var: &OutputVariables,
) -> Result<bool, Box<dyn Error>> {
    let mut f = File::create(output_file)?;

    write!(f, "NODES_PLACEMENT = [\n")?;
    for i in 0..in_var.num_nodes {
        write!(
            f,
            "{} {} {}\n",
            i, out_var.node_pos_x[i as usize], out_var.node_pos_y[i as usize]
        )?;
    }
    write!(f, "];\n\n")?;
    write!(f, "PRR_MATRIX = [ \n")?;
    for i in 0..in_var.num_nodes {
        for j in 0..in_var.num_nodes {
            write!(f, "{}  ", out_var.gen[i as usize][j as usize])?;
        }
        write!(f, "\n")?;
    }
    write!(f, "];\n\n")?;
    write!(f, "PR_MATRIX = [ \n")?;
    for i in 0..in_var.num_nodes {
        for j in 0..in_var.num_nodes {
            write!(f, "{}  ", out_var.pr[i as usize][j as usize])?;
        }
        write!(f, "\n")?;
    }
    write!(f, "];\n")?;

    Ok(true)
}

fn get_input() -> Result<InputVariables, confy::ConfyError> {
    let _cfg: Result<InputVariables, confy::ConfyError> = confy::load_path("topology.conf");
    return _cfg;
}

fn main() {
    if !Path::new("topology.conf").exists() {
        println!("Config not present in work place.");
        std::process::exit(1);
    }

    let mut in_var: InputVariables;
    // input parameters use in_var
    if let Ok(i_c) = get_input() {
        // ... use config ...

        in_var = i_c;
    } else {
        // ... config is not available, may be should
        // we warn the user, ask for an alternative ...
        println!("Couldn't get config.");
        std::process::exit(1);
    }
    // parse input file
    //read_file("inputFile.m", &mut in_var);
    // output parameters use out_var
    let mut out_var = OutputVariables::new(in_var.num_nodes.try_into().unwrap());
    // create topology
    println!("Topology for {} nodes ...\t\t\t", in_var.num_nodes);
    if obtain_topology(&in_var, &mut out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    // obtain output power and noise floor
    print!("Radio Pt and Pn ...\t\t");
    if obtain_radio_pt_pn(&in_var, &mut out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    // based on distance, obtain rssi for all the links
    print!("Received Signal Strength ...\t");
    if obtain_rssi(&in_var, &mut out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    // based on rssi, obtain prob. of error for all the links
    print!("Probability of Error ...\t");
    if obtain_prob_error(&in_var, &mut out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    // based on prob. of error, obtain packet reception rate for all the links
    print!("Packet Reception Rate ...\t");
    if obtain_prr(&in_var, &mut out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    // provide Matrix result
    print!("Printing Output File ...\t");
    if let Ok(temp) = print_file("outputFile", &in_var, &out_var) {
        println!("done");
    } else {
        std::process::exit(1);
    }
    println!("done");
}
