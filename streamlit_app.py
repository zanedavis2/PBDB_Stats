824
825
826
827
828
829
830
831
832
833
834
835
836
837
838
839
840
841
842
843
844
845
846
847
848
849
850
851
852
853
854
855
856
857
858
859
860
861
862
863
864
865
866
867
868
869
870
871
872
873
874
875
876
877
878
879
880
881
882
883
884
885
886
import os
    for c in cols:
        try:
            out[c] = out[c].apply(lambda x: (f"{x:.3f}" if pd.notna(x) else "")).str.replace("0.", ".", regex=False)
        except Exception:
            pass
    return out

# ---- Hitting ----
with T1:
    if hitting_df is None:
        st.info("No hitting data yet.")
    else:
        hd = hitting_df.copy()
        if "PA" in hd.columns:
            hd = hd[hd["PA"] >= min_pa]
        hd = _apply_common_filters(hd, name_filter)
        hd = _fmt_rates(hd, "Hitting")
        st.dataframe(hd, use_container_width=True)
        st.download_button("Download Hitting CSV", data=hd.to_csv(index=False).encode("utf-8"), file_name="hitting.csv")
        with st.expander("Acronym key"):
            st.dataframe(HITTING_KEY, use_container_width=True)

# ---- Pitching ----
with T2:
    if pitching_df is None:
        st.info("No pitching data yet.")
    else:
        pdx = pitching_df.copy()
        if "IP" in pdx.columns:
            pdx = pdx[pdx["IP"] >= min_ip]
        pdx = _apply_common_filters(pdx, name_filter)
        pdx = _fmt_rates(pdx, "Pitching")
        st.dataframe(pdx, use_container_width=True)
        st.download_button("Download Pitching CSV", data=pdx.to_csv(index=False).encode("utf-8"), file_name="pitching.csv")
        with st.expander("Acronym key"):
            st.dataframe(PITCHING_KEY, use_container_width=True)

# ---- Fielding ----
with T3:
    if fielding_df is None:
        st.info("No fielding data yet.")
    else:
        fd = fielding_df.copy()
        if "TC" in fd.columns:
            fd = fd[fd["TC"] >= min_tc]
        fd = _apply_common_filters(fd, name_filter)
        st.dataframe(fd, use_container_width=True)
        st.download_button("Download Fielding CSV", data=fd.to_csv(index=False).encode("utf-8"), file_name="fielding.csv")

# ---- Catching ----
with T4:
    if catching_df is None:
        st.info("No catching data yet.")
    else:
        cd = catching_df.copy()
        if "INN" in cd.columns:
            cd = cd[cd["INN"] >= min_inn]
        cd = _apply_common_filters(cd, name_filter)
        st.dataframe(cd, use_container_width=True)
        st.download_button("Download Catching CSV", data=cd.to_csv(index=False).encode("utf-8"), file_name="catching.csv")

st.caption("Tip: In 'Select series' mode, ensure your CSV files are named exactly as '<series>.csv' in the working directory.")
