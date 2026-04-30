import pandas as pd

def render_score_calculator(
    col_result,
    predict,
    income,
    age,
    loan_amount,
    loan_tenure,
    existing_loans,
    on_time_pct,
    credit_util,
    savings_pct,
    enquiries,
    employment,
    model_ready,
    all_data,
    FEATURE_NAMES,
    CreditNet,
    prob_to_cibil,
    score_label,
    icon_status,
    st,
    np,
    go,
    torch,
    StandardScaler
):
    with col_result:
        st.subheader("Score Output")

        if predict:
            emp_stability = {
                "Government (Salaried)": 0.95,
                "Private (Salaried)": 0.75,
                "Self-Employed": 0.55,
                "Business Owner": 0.65,
                "Freelancer": 0.40,
                "Unemployed": 0.10,
            }

            # CIBIL score uses only credit-related factors
            features_for_score = np.array([[
                income,
                age,
                loan_amount,
                loan_tenure,
                existing_loans,
                on_time_pct / 100,
                credit_util / 100,
                0.5, 
                savings_pct / 100,
                enquiries,
            ]], dtype=np.float32)

            if model_ready:
                model = st.session_state["trained_model"]

                if "scalers" in st.session_state:
                    scaler = list(st.session_state["scalers"].values())[0]
                else:
                    scaler = StandardScaler()
                    scaler.fit(all_data["HDFC"][FEATURE_NAMES].values)

                X_sc = scaler.transform(features_for_score).astype(np.float32)

            else:
                model = CreditNet(10)
                scaler = StandardScaler()
                scaler.fit(all_data["HDFC"][FEATURE_NAMES].values)
                X_sc = scaler.transform(features_for_score).astype(np.float32)

            model.eval()

            with torch.no_grad():
                prob = model(torch.from_numpy(X_sc)).item()

            score = prob_to_cibil(prob)
            label, clr = score_label(score)

            emi = loan_amount / loan_tenure * (1 + 0.10 / 12) ** loan_tenure
            dti = (emi / income) * 100

            emp_score = emp_stability[employment]

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={
                        "text": f"<b>CIBIL Score</b> — <span style='color:{clr}'>{label}</span>"
                    },
                    gauge={
                        "axis": {"range": [300, 900]},
                        "bar": {"color": clr},
                        "steps": [
                            {"range": [300, 550], "color": "#1a0a0a"},
                            {"range": [550, 650], "color": "#1a1a0a"},
                            {"range": [650, 750], "color": "#0a1a0a"},
                            {"range": [750, 900], "color": "#0a1a10"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": 750,
                        },
                    },
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(t=60, b=10)
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="cp_gauge"
            )

            if score >= 750 and emp_score >= 0.5:
                icon_status(
                    "fa-circle-check",
                    "Loan likely APPROVED — Excellent credit & stable employment",
                    "#22c55e",
                    "#052e16"
                )

            elif score >= 750 and emp_score < 0.5:
                icon_status(
                    "fa-triangle-exclamation",
                    "Conditional approval — Excellent credit but employment concern",
                    "#f97316",
                    "#1c0a00"
                )

            elif score >= 650 and emp_score >= 0.5:
                icon_status(
                    "fa-circle-check",
                    "Loan likely APPROVED — Good credit & stable employment",
                    "#22c55e",
                    "#052e16"
                )

            elif score >= 650 and emp_score < 0.5:
                icon_status(
                    "fa-triangle-exclamation",
                    "Conditional approval — Good credit but employment concern",
                    "#f97316",
                    "#1c0a00"
                )

            elif score >= 550:
                icon_status(
                    "fa-triangle-exclamation",
                    "Conditional approval — Fair credit",
                    "#f97316",
                    "#1c0a00"
                )

            else:
                icon_status(
                    "fa-circle-xmark",
                    "Loan likely REJECTED — Poor credit",
                    "#ef4444",
                    "#1a0000"
                )

            st.divider()

            m1, m2 = st.columns(2)

            m1.metric("CIBIL Score", score)
            m1.metric("Default Probability", f"{prob:.2%}")

            m2.metric("Suggested EMI", f"INR {emi:,.0f}/month")
            m2.metric(
                "Debt-to-Income Ratio",
                f"{dti:.1f}%",
                delta="Good" if dti < 40 else "High",
                delta_color="normal" if dti < 40 else "inverse"
            )

            st.markdown(
                f"**Employment Status:** {employment} "
                f"(Stability Score: {emp_score:.2f})"
            )

            st.caption(
                "Employment affects loan approval decision but not CIBIL score calculation."
            )

            last_mode = st.session_state.get("last_mode", "unknown")

            st.caption(
                f"Predicted using "
                f"{'FL-trained model (' + last_mode + ')' if model_ready else 'untrained demo model'}. "
                f"Raw bank data was never shared."
            )

        else:
            st.info(
                "Fill in the customer details and click **Predict Credit Score**."
            )

            st.dataframe(
                pd.DataFrame({
                    "Score": ["750-900", "650-749", "550-649", "300-549"],
                    "Rating": ["Excellent", "Good", "Fair", "Poor"],
                    "Decision": ["Approved", "Approved", "Conditional", "Rejected"],
                    "Interest Rate": ["8-10%", "10-13%", "13-18%", "18%+"],
                }),
                use_container_width=True,
                hide_index=True
            )
