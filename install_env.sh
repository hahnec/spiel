# create environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
git clone --recurse-submodules git@github.com:hahnec/multimodal_emg ./multimodal_emg_repo

# checkout dependency repo at commit
cd multimodal_emg_repo
git reset --hard e111a230dfa5ff75adf2fe6422209cb7c65bb1a0
python3 -m pip install -r requirements.txt
cd ..

# make link for module level access
ln -sf ./multimodal_emg_repo/multimodal_emg ./multimodal_emg
